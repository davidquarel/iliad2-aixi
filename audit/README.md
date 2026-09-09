# [2.6] Goal misgeneralisation notebook: problems found and fixes made

Audit trail for branch `goalmisgen-2.6-benchmark` (base: `main` at `7b715a6`). Companion files:

- `master_2_6.annotated.py`: the current master with an `AUDIT` block before every cell that
  changed, quoting the original and giving the reason and evidence. Not read by the generator.
- `ppo.annotated.py`: the current trainer with an `AUDIT` block before every changed function.
- Full data and tables: `../goalmisgen-benchmark-report.md` (first pass) and
  `../goalmisgen-sweep-report.md` (overnight sweep).

To see the raw diff: `git diff main -- gen/masters/master_2_6.py gen/support/part6_goalmisgen/ppo.py`.

## Problems, in order of severity

### P1. net4 (the "fix" for goal misgeneralisation) failed about half the time

The notebook trains net4 on `generate_shift` for 4096 steps and claims reward2 recovers. Replicating
with the notebook's own seed, net4 binned nothing in 1000 shifted layouts. Repeating the identical
configuration gave 0%, 0%, 85%, 99%, 99% binning across five runs: fixed seeds do not fix the
outcome because GPU kernels are nondeterministic and the dynamics are chaotic. Training curves show
a plateau near return 0.15 ("pick up and hold") for 2500 to 3500 steps followed by a sharp takeoff,
so a 4096-step budget ends right at the edge. The original JAX lab used 10240 steps.

**Root cause:** the trainer did one gradient update per batch of rollouts (32 environments), so
learning was extremely sample-inefficient and the exploration plateau was long.

**Fix:** multi-epoch minibatch updates (4 epochs x 4 minibatches), 512 parallel environments,
lr 3e-3, entropy 0.01. Takeoff now happens at step ~110 to 160 on every seed tried (sweeps 1, 3, 4;
validation on 5 seeds), net4 reaches 97% of the optimal return in 192 steps (~25 s on an A4000).

### P2. net2 (the "fixed specification" agent) only cleaned one pile of four

The notebook admits this ("only reliably cleans up one pile"). Against an exact planner the shipped
net2 reached 24% of the optimal return; net1 reached 28% of the farming optimum. The lesson the
section is meant to teach (a well-specified reward produces the intended behaviour) was being
demonstrated by an agent that barely did the task.

**Root cause:** same single-update trainer, plus lr 1e-3. In sweep 2, single-update configurations
never exceed 27% (net2) or 78% (net1) in 1024 steps; multi-epoch with lr 1e-3 plateaus at 78%
(three piles).

**Fix:** 8 epochs x 4 minibatches, 256 rollouts, lr 3e-3. net2 reaches 99.8% (all four piles) in
192 steps, net1 reaches 99.5% of the farming optimum in 96 steps.

### P3. net2 had a hidden exploration trap at three piles

With the faster trainer, about one seed in four got stuck at exactly 77% of optimum (three piles)
and stayed there to 512 steps, regardless of batch size or minibatch structure and for entropy
0.001 to 0.01 (sweeps 4b, 5). After binning three piles the near-deterministic policy never explores
its way to the far fourth pile.

**Fix:** entropy coefficient 0.03 for the fixed-layout trainer. 8 of 8 seeds then reach >= 95% (worst
128 steps) with drop and break probes still near zero; 0.1 is too noisy; annealing schedules were no
better than the constant; a bigger network also works but would change student-visible code (sweeps
6, 7). net1 still farms at 95 to 97% of optimum at 0.03.

### P4. Two exercise templates rendered fully commented out in the student notebook

The generator un-comments an `EXERCISE` block only if every line starts with `# `; the `env2` and
`env_shift`/`proxy` templates contained a bare `#` line, so students got the whole template as a
comment and the `proxy` cell then raised `NameError` from its test call.

**Fix:** removed the bare `#` lines. Verified in the regenerated exercises notebook.

### P5. Training was slow for structural reasons

Per-step cost is launch-bound: 64 sequential environment steps of tiny kernels plus Python
overhead, so the GPU idles and a 1024-environment batch costs only ~1.3x a 32-environment batch.

**Fixes:** (a) large batches; (b) rollouts collected by a `torch.compile(mode="reduce-overhead")`
CUDA-graph step, 6 to 8x faster per rollout, bit-identical environment dynamics, eager fallback on
CPU or compile failure; (c) on CPU, cap PyTorch threads at 4 (a 48-core box was 9x slower with the
default). Whole notebook: ~16 min (net4 unreliable) -> 96 s, all four agents at >= 97% of optimum.

### P6. Student errata, checked

- "`reward_shaped` should halve the shaping term": not legit. The shipped code is potential
  shaping with Phi = 0.5 as the hint prescribes; halving is also valid shaping but fails the
  sheet's own test. No change.
- "Nothing goes on the GPU": not literally true (all tensors are on `cuda:0` when a GPU exists).
  What students hit is the Colab default runtime having no GPU, and the notebook only mentioning
  runtime type in section 4. Fix: GPU-runtime note at the top; stray "or TPU" removed.

### P7. Smaller things

- A leaked `#CLAUDE:` bookkeeping comment was visible in both student notebooks. Removed.
- PPO's clipping was inert with one update per batch (ratio exactly 1). Now meaningful.
- The bonus "tune the agent" bullet assumed net2 cleans one pile. Replaced by "break the agent":
  students weaken the optimiser to reproduce the old behaviour.
- The bonus "everyone has a price" claims (PPO learns nothing on the wall layouts) were re-checked
  with the stronger trainer and still hold. No change.

## What was deliberately not changed

Environment dynamics, reward functions and constants (Phi = 0.5, +1 bin, -2 urn), network
architectures, discount 0.995, 64-step horizon, the pinned-bin/shifted-bin set-up, all tests, and
every exercise the students write. The RL algorithm is treated as a black box for this day, so all
trainer changes live in `ppo.py`; the notebook's training loops got shorter, not longer.

## Reproducibility notes

- The compiled rollout path seeds the device RNG from the notebook's generator each rollout; two
  end-to-end runs on different GPUs gave bit-identical results.
- Not verified on Colab itself. Compile needs triton (present on Colab GPU runtimes); any failure
  prints a note and falls back to eager rollouts. CPU-only takes ~10 min for the whole notebook.
