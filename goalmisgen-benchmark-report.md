# [2.6] Goal Misgeneralisation: errata check, benchmark, and trainer speedup

Progress report, 2026-09-09. Scope: the notebook (`gen/masters/master_2_6.py`) and its
support package (`gen/support/part6_goalmisgen/`). Slides not touched. All runs on a
local NVIDIA A40 in the ARENA venv; the box is shared, so wall-clock figures marked
"(under load)" were measured while other users' jobs were running and are pessimistic.

## Student errata

**(a) "`reward_shaped` should return `bin_reward_term + pickup_shaping_term / 2`" — not legit.**
The shipped code is potential shaping with Φ(s) = 0.5 while holding shards, exactly as the
hint prescribes. A pickup-then-floor-drop cycle nets exactly zero discounted return either
way; dividing the shaping term by 2 is just Φ = 0.25, which is also valid shaping but fails
the sheet's own `test_reward_shaped` (it pins pickup = +0.4975). No change.

| variant | pickup | floor drop | bin drop | pickup→drop cycle |
|---|---|---|---|---|
| shipped (Φ = 0.5) | +0.4975 | −0.5 | +0.5 | 0.000 |
| student's /2 (Φ = 0.25) | +0.2488 | −0.25 | +0.75 | 0.000 |

**(b) "Nothing ever gets put on the GPU; training is on CPU by default" — not legit as stated.**
Instrumenting the reward function during training shows every transition and all parameters
on `cuda:0` in both training loops whenever a GPU is available. Two real things sit behind
the perception:

1. On Colab the default runtime has no GPU, and the notebook only said to switch runtime type
   in section 4. Fixed: GPU-runtime note at the top of the setup section; stray "or TPU" removed
   (the code only detects CUDA/MPS).
2. On CPU, PyTorch's default thread pool (one thread per core) makes this workload much
   slower on many-core machines, because each rollout is 64 sequential steps on tiny tensors
   and every op pays fork/join overhead. Irrelevant on Colab (2 vCPUs), but a 48-core box
   trained 9x slower than with 4 threads. Fixed: cap at 4 threads when on CPU (no effect on GPU).

| CPU threads | `train_agent` ms/step | `train_agent_multienv` ms/step |
|---|---|---|
| 1 | 66 | 139 |
| 4 | 50 | 89 |
| 16 | 144 | 231 |
| 48 (default) | 588 | 856 |
| GPU, any thread count | 87 | 105 |

## Replication of the notebook as shipped (GPU, notebook's own seeds)

| agent | steps | time | result |
|---|---|---|---|
| net1 (`reward1`, fixed layout) | 256 | 50 s | farms floor-drops (drop probe 7.2), never bins. As claimed. |
| net2 (`reward2`, fixed layout) | 512 | 45 s | bins (0.85), break probe 0.00, drop probe 0 in 80% of rollouts. As claimed. |
| net3 (`generate`, bin pinned) | 4096 | 7.1 min | bins 99% in-distribution; on `env_shift` bins 0%, proxy 1.12. As claimed. |
| net4 (`generate_shift`) | 4096 | 6.8 min | **bins 0% of 1000 shifted layouts; 85% of rollouts return exactly 0. Claim fails.** |

The bonus "everyone has a price" numbers reproduce exactly (detour +1.189 vs smash +1.356;
wall layout 0 vs +1.416).

### Why net4 failed, and why it is a coin flip

Repeating net4's exact configuration gave different outcomes on different runs even with
the same seeds (GPU kernels are nondeterministic and the dynamics are chaotic). Training
curves show a plateau near return 0.15 ("pick up and hold") for 2500–3500 steps, then a sharp
takeoff. A 4096-step budget ends right at that edge.

| config (shifted task) | runs | bins on 1000 shifted layouts |
|---|---|---|
| 4096 steps, entropy 0.01 (shipped) | 5 | 0%, 0%, 85%, 99%, 99% |
| 4096 steps, entropy 0.001 (original JAX lab's value) | 3 | 0%, 0%, 98% (one run collapsed to a deterministic policy) |
| 8192 steps, entropy 0.01 | 3 | 99%, 99%, 100% |
| 12288 steps, entropy 0.01 | 1 | 99% |

Lower entropy is worse. The original lab used 10240 steps for this agent. An interim fix
(8192 steps, ≈14 min) was applied, then superseded by the trainer change below.

## Trainer speedup

Per-step cost is dominated by kernel-launch and Python overhead in the 64-step rollout loop,
not compute (the GPU is no faster than a 4-thread CPU). So the levers are reusing each batch
of rollouts for more updates, and collecting more environments per launch.

`ppo.py` now takes `num_epochs` / `num_minibatches` (defaults 1/1 reproduce the old trainer
bit-for-bit; with epochs > 1 the PPO clipping actually engages — it was inert before, since a
single update per batch has ratio exactly 1). Benchmarked on the shifted task; "takeoff" is the
first step at which the 32-step mean training return exceeds 2.0.

| config | takeoff step (seeds) | transitions to takeoff | shifted-layout binning after run |
|---|---|---|---|
| 32 envs, 1 update (shipped) | 2600–3600, or never | 5–7M | 0–99% |
| 32 envs, 4 epochs × 4 minibatches | 528, 374 | 0.8–1.1M | 100%, 99% |
| 128 envs, 1 update | 1610, 2012 | 13–16M | 99%, 96% |
| **128 envs, 4 × 4** | **211, 196, 200** | **1.6–1.7M** | **100%** |
| 128 envs, 4 × 4, lr 3e-3 | 139 | 1.1M | not evaluated |
| 128 envs, 4 × 4, pinned bin (net3 role) | 77 | 0.6M | 7% (still misgeneralises: proxy 0.78) |

Isolated timed runs of the chosen config (under load):

| run | steps | wall | takeoff | result |
|---|---|---|---|---|
| net3 role (pinned), 128 envs 4×4 | 512 | 4.3 min | 35 s | bins 7% on shift, 0% on `env_shift`, proxy 1.0 — misgeneralises |
| net4 role (shift), 128 envs 4×4 | 512 | 4.0 min | 107 s | bins 100% on shift, proxy 0.00 on `env_shift` |
| net4 role (shift), 32 envs 4×4 | 1024 | 8.3 min | 244 s | bins 99% |

Net effect for the notebook: net3 and net4 go from ~7 min (net4 unreliable) to ~4 min each
with a reliable outcome. net3 exits the plateau by step ~80, so it could be shortened further.

## Changes made (uncommitted before this branch)

`gen/masters/master_2_6.py`
- Setup: GPU-runtime note; CPU thread cap; "or TPU" removed.
- Two exercise templates (`env2`; `env_shift`/`proxy`) had a bare `#` line, which stops the
  generator un-commenting the template, so the student notebook shipped them fully commented
  and the `proxy` cell then raised NameError from its test call. Fixed.
- Leaked `#CLAUDE:` bookkeeping comment removed from a HIDE block.
- `train_agent_multienv`: 128 envs per step, `num_epochs=4, num_minibatches=4`.
- net3 and net4: 4096 → 512 steps (vis every 16). Prose describing the trainer, the training
  time, and the net4 plateau updated accordingly.

`gen/support/part6_goalmisgen/ppo.py`
- Multi-epoch minibatch updates (see above). `ppo_loss_fn` now takes flat transitions.
  Package regression tests pass; defaults verified bit-identical to the old trainer.

## Reported, not changed

- With one update per batch, PPO's `proximity_eps` had no effect (clip fraction 0, KL ~1e-9).
  Now meaningful in the multi-env trainer; still inert in `train_agent` (net1/net2), which
  was left as is since those runs are already ~1 min.
- Fixed 64-step horizon with no terminal potential correction leaves a small bonus (~0.36)
  for holding shards at episode end. Harmless (binning always pays more) but it is the local
  optimum the plateau sits on, visible as reward2 ≈ +0.363 for misgeneralising agents.
- The `randperm` minibatch shuffle uses the global RNG, so CPU runs are no longer bit-reproducible
  from the seed alone (GPU runs never were).

## Open / in progress

- End-to-end run of the regenerated `solutions.py` (all four trainings, new trainer) was in
  progress when this report was written, with all tests passing through section 5 and net4
  training. Final evaluation numbers to be appended.
- Options discussed but not taken: shipping pretrained checkpoints for net3/net4 (guarantees
  the demo; user considering), lower γ, entropy schedule, compiling the env step.
- Not verified on Colab itself.

## Reproducing

```
PYTHONPATH=gen/support python3 gen/support/part6_goalmisgen/tests.py   # package tests
python3 gen/core/main.py --chapters=2.6 --use_py=true                   # regenerate notebooks
```
Benchmark scripts lived in the session scratchpad and are not committed.
