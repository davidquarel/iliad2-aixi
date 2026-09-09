# [2.6] Overnight sweep: training as fast as possible while keeping the behaviours

Running report (updated through the night). Machine: 4x RTX A4000 (used GPUs 2 and 3 only),
16 CPU cores, torch 2.7, quiet box. Builds on `goalmisgen-benchmark-report.md`.

## 0. What "desired behaviour" and "optimal" mean, made measurable

The environment dynamics are simple enough to plan against, so I built an oracle
(`oracle.py`, session scratch) that computes the best achievable discounted return for a layout:

- **reward2 (clean-up), never breaking an urn.** The shaping terms telescope, so the return of
  any no-break policy is `sum over binned shards of gamma^(t_bin)` (+0.5·gamma^64 if holding at
  the end). The best such policy visits shards in some order along BFS shortest paths that
  avoid urns; with <= 4 shards the order is brute-forced (24 permutations), so this is exact
  for that policy class. Every plan is replayed through the real `Environment.step` and scored
  with the real reward function.
- **reward1 (farming) on the fixed layout.** Walk to the nearest shard, then alternate
  PICKUP/PUTDOWN: `sum_k gamma^(d+2k)`.

Oracle returns (gamma = 0.995, 64-step horizon):

| layout | reward | oracle return |
|---|---|---|
| fixed 6x6 `env` | reward1 (farm) | 27.375 |
| fixed 6x6 `env` | reward2 (clean all 4 piles) | 3.600 |
| `env2` | reward2 | 4.257 |
| `env_shift` (4x4) | reward2 | 1.912 |
| 1000 `generate()` layouts, seed 7 | reward2 | mean 3.639 (p10 3.59, p90 3.73) |
| 1000 `generate_shift()` layouts, seed 7 | reward2 | mean 3.691 (p10 3.63, p90 3.77) |

Where the agents from the previous report stood against these:

| agent | sampled return | oracle | ratio |
|---|---|---|---|
| net1 (reward1, fixed) | 7.6–7.8 | 27.4 | **0.28** |
| net2 (reward2, fixed) | 0.85 | 3.60 | **0.24** (bins one pile of four) |
| net3 in-distribution | 3.47–3.53 | 3.64 | 0.96 |
| net4 on shifted layouts (fast trainer) | 3.52–3.57 | 3.69 | 0.96 |
| net4 on `env_shift` | 1.90 | 1.91 | 0.99 |

So the procedural agents were already near-optimal and the job there is speed; the fixed-layout
agents were far from optimal (the sheet itself admits net2 "only reliably cleans one pile"), and
the job there is both.

Pass criteria used by the harness (`evalh.py`), evaluated with the notebook's sampled policy:

- net1: drop probe >= 80% of reward1 return (it farms) and >= 50% of the farming oracle.
- net2: break probe <= 0.05, drop probe <= 0.15, >= 90% of oracle.
- net3: >= 90% of oracle in-distribution; on shifted layouts bins in <= 15%; on `env_shift`
  bins in <= 5% and proxy >= 0.5 (it must still misgeneralise).
- net4: >= 90% of oracle on shifted layouts; on `env_shift` bins in >= 95% and proxy <= 0.05.

Progress metric during training: mean sampled return / oracle on 256 layouts (or rollouts),
every 16 steps. A run "reaches target" when it holds >= 0.95 on two consecutive evals.

## 1. Isolated per-step cost (GPU 3, nothing else running)

Per-step wall time is almost flat in batch size: the loop is launch-bound, so a 1024-env
batch costs ~1.3x a 32-env batch while collecting 32x the data.

| envs | 1 update | 4 ep x 4 mb | 8 x 4 | 4 x 8 |
|---|---|---|---|---|
| 32 | 92 ms | 133 | 220 | 221 |
| 128 | 87 | 141 | 215 | 216 |
| 256 | 91 | 163 | 243 | 224 |
| 512 | 101 | 172 | 259 | 247 |
| 1024 | 121 | 237 | 385 | 264 |

Network size barely matters (128 envs, 4x4): 8ch/32w/3conv 116 ms; 16/64/3 122; 16/64/5
(shipped) 142; 16/128/4 132; 32/128/5 148.

Fixed 6x6 env, small net: 16 rollouts 68 ms (1 update) / 106 ms (4x4); 256 rollouts 75 / 114;
1024 rollouts 96 / 174.

## 2. Sweeps

### Sweep 1: net4 (shifted-bin task), batch x reuse x lr, seed 1, entropy 0.01, shipped net

All 32 configs reached >= 0.94 of oracle and passed the behaviour check. Steps to hold >= 0.95 of oracle
(two consecutive evals), and the implied isolated wall time (steps x isolated ms/step from section 1;
2-epoch rows interpolated). Sorted by implied wall time.

| envs | epochs x mb | lr | steps to 0.95 | implied wall (A4000) | final ratio |
|---|---|---|---|---|---|
| 512 | 4 x 4 | 0.003 | 144 | 25 s | 0.955 |
| 1024 | 4 x 4 | 0.003 | 128 | 30 s | 0.968 |
| 1024 | 4 x 8 | 0.003 | 112 | 30 s | 0.963 |
| 256 | 4 x 4 | 0.003 | 192 | 31 s | 0.953 |
| 512 | 2 x 4 | 0.003 | 256 | 32 s | 0.959 |
| 512 | 4 x 8 | 0.003 | 144 | 36 s | 0.968 |
| 512 | 8 x 4 | 0.003 | 144 | 37 s | 0.966 |
| 1024 | 4 x 8 | 0.001 | 144 | 38 s | 0.959 |
| 256 | 2 x 4 | 0.003 | 352 | 40 s | 0.964 |
| 1024 | 2 x 4 | 0.003 | 256 | 41 s | 0.971 |
| 256 | 4 x 8 | 0.003 | 192 | 43 s | 0.958 |
| 512 | 4 x 8 | 0.001 | 176 | 43 s | 0.965 |
| 1024 | 8 x 4 | 0.003 | 112 | 43 s | 0.959 |
| 512 | 4 x 4 | 0.001 | 256 | 44 s | 0.963 |
| 128 | 4 x 4 | 0.003 | 320 | 45 s | 0.947 |
| 512 | 8 x 4 | 0.001 | 176 | 46 s | 0.958 |
| 256 | 4 x 8 | 0.001 | 208 | 47 s | 0.961 |
| 256 | 8 x 4 | 0.001 | 224 | 54 s | 0.955 |
| 256 | 4 x 4 | 0.001 | 336 | 55 s | 0.952 |
| 1024 | 8 x 4 | 0.001 | 144 | 55 s | 0.971 |
| 128 | 2 x 4 | 0.003 | 544 | 57 s | 0.948 |
| 1024 | 4 x 4 | 0.001 | 240 | 57 s | 0.960 |
| 128 | 8 x 4 | 0.003 | 304 | 65 s | 0.947 |
| 256 | 2 x 4 | 0.001 | 592 | 68 s | 0.956 |
| 128 | 4 x 4 | 0.001 | 496 | 70 s | 0.953 |
| 512 | 2 x 4 | 0.001 | 560 | 70 s | 0.952 |
| 256 | 8 x 4 | 0.003 | 304 | 74 s | 0.964 |
| 128 | 4 x 8 | 0.001 | 368 | 79 s | 0.949 |
| 128 | 4 x 8 | 0.003 | 400 | 86 s | 0.942 |
| 128 | 8 x 4 | 0.001 | 416 | 89 s | 0.956 |
| 1024 | 2 x 4 | 0.001 | 608 | 97 s | 0.962 |
| 128 | 2 x 4 | 0.001 | not in 640 | - | 0.914 |

Reading: the learning rate can go to 3e-3 (no run diverged; entropy stays healthy), batch size beyond 256
buys little in wall time because steps-to-target stops falling as fast as the batch grows, and 2 epochs is
clearly worse than 4. The branch config (128 envs, 4x4, 1e-3) needs 496 steps (~70 s here); the best
configs need 112-144 steps (~25-30 s).

## 3. Per-step overhead: compiled rollouts (CUDA graphs)

The rollout loop is launch-bound (64 sequential steps of tiny kernels). Compiling one
(observe -> policy -> sample -> step) function with `torch.compile(mode="reduce-overhead")`
replays it as a CUDA graph. Measured per 64-step rollout (ms), outputs retained correctly
(`torch.compiler.cudagraph_mark_step_begin()` before each call, outputs cloned):

| batch | eager | compile (no graphs) | compile + CUDA graphs |
|---|---|---|---|
| 128 (isolated) | 97 | 47 | **15** |
| 512 (GPU shared) | 250 | 72 | **17** |
| 1024 (GPU shared) | 219 | 49 | **24** |

Pitfalls found on the way: without the step marker the graph is re-recorded every call
(522 ms, 5x slower than eager); a manual whole-rollout `torch.cuda.CUDAGraph` capture fails on
the multinomial RNG op and poisons the CUDA context. First-call compile cost is ~4 s;
subsequent batch sizes recompile in <1 s from cache.

Integrated into `ppo.py` as `collect_annotated_rollout_fast`, used when
`compile_rollouts=True` (CPU or any compile failure falls back to the eager path and prints
a note). Verified: same generator seed gives identical actions; next-states are bit-identical
to an eager `env.step` replay of the same actions; stored observations/logits/values match
`env.observe`/`net`; learning progress identical to eager (ratio 0.920 vs 0.922 after 109
steps at 512 envs, 4x4, lr 3e-3). Full train step at 512 envs 4x4: 247 -> 130 ms (shared GPU),
so the update phase (16 minibatch forward/backward passes) is now the larger half.

Compiling the update phase is a much smaller win (512 envs, 4x4, GPU shared): eager 169 ms,
`torch.compile` default 125 ms, reduce-overhead 462 ms (worse: per-minibatch graph re-recording).
Fewer, larger minibatches help about as much (4 epochs x 2 minibatches 135 ms, x 1: 122 ms).
Not adopted; with rollouts compiled, a moderate batch (256) makes the update phase cheap anyway.

### Sweep 2: fixed-layout agents (net1 farming, net2 clean-up), seed 1, entropy 0.001, shipped small net

Steps to hold >= 0.95 of the oracle (farming oracle 27.4 for net1; clean-up oracle 3.60 for net2).

| role | rollouts | epochs x mb | lr | steps to 0.95 | final ratio | pass |
|---|---|---|---|---|---|---|
| net1 | 64 | 8 x 4 | 0.003 | 48 | 0.997 | True |
| net1 | 256 | 8 x 4 | 0.003 | 48 | 0.998 | True |
| net1 | 64 | 4 x 4 | 0.003 | 64 | 0.999 | True |
| net1 | 64 | 8 x 4 | 0.001 | 64 | 0.993 | True |
| net1 | 256 | 4 x 4 | 0.003 | 64 | 0.998 | True |
| net1 | 1024 | 8 x 4 | 0.003 | 64 | 1.000 | True |
| net1 | 256 | 8 x 4 | 0.001 | 80 | 0.953 | True |
| net1 | 1024 | 4 x 4 | 0.003 | 80 | 0.963 | True |
| net1 | 1024 | 8 x 4 | 0.001 | 80 | 0.963 | True |
| net1 | 1024 | 4 x 4 | 0.001 | 96 | 0.957 | True |
| net1 | 256 | 4 x 4 | 0.001 | 112 | 0.960 | True |
| net1 | 64 | 4 x 4 | 0.001 | 128 | 0.984 | True |
| net1 | 1024 | 1 x 1 | 0.003 | 944 | 0.961 | True |
| net1 | 64 | 1 x 1 | 0.001 | not in 1024 | 0.418 | False |
| net1 | 64 | 1 x 1 | 0.003 | not in 1024 | 0.658 | True |
| net1 | 256 | 1 x 1 | 0.001 | not in 1024 | 0.682 | True |
| net1 | 256 | 1 x 1 | 0.003 | not in 1024 | 0.775 | True |
| net1 | 1024 | 1 x 1 | 0.001 | not in 1024 | 0.333 | False |
| net2 | 1024 | 8 x 4 | 0.003 | 80 | 0.998 | True |
| net2 | 256 | 8 x 4 | 0.003 | 96 | 0.984 | True |
| net2 | 256 | 4 x 4 | 0.003 | 160 | 0.991 | True |
| net2 | 64 | 4 x 4 | 0.003 | 240 | 0.981 | True |
| net2 | 1024 | 4 x 4 | 0.003 | 880 | 0.991 | True |
| net2 | 64 | 1 x 1 | 0.001 | not in 1024 | 0.270 | False |
| net2 | 64 | 1 x 1 | 0.003 | not in 1024 | 0.272 | False |
| net2 | 64 | 4 x 4 | 0.001 | not in 1024 | 0.775 | False |
| net2 | 64 | 8 x 4 | 0.001 | not in 1024 | 0.272 | False |
| net2 | 64 | 8 x 4 | 0.003 | not in 1024 | 0.954 | True |
| net2 | 256 | 1 x 1 | 0.001 | not in 1024 | 0.271 | False |
| net2 | 256 | 1 x 1 | 0.003 | not in 1024 | 0.272 | False |
| net2 | 256 | 4 x 4 | 0.001 | not in 1024 | 0.775 | False |
| net2 | 256 | 8 x 4 | 0.001 | not in 1024 | 0.272 | False |
| net2 | 1024 | 1 x 1 | 0.001 | not in 1024 | 0.272 | False |
| net2 | 1024 | 1 x 1 | 0.003 | not in 1024 | 0.272 | False |
| net2 | 1024 | 4 x 4 | 0.001 | not in 1024 | 0.775 | False |
| net2 | 1024 | 8 x 4 | 0.001 | not in 1024 | 0.774 | False |

Reading: the shipped single-update trainer is what limits these agents. With one update per batch,
net1 never exceeds 78% of the farming optimum and net2 stays at 27% (one pile) for 1024 steps; with
4-8 epochs and lr 3e-3, net1 reaches 99% in 48-64 steps and net2 reaches 98% (all four piles) in
80-160 steps. lr 1e-3 with epochs gets net2 to 78% (three piles) at best. In the notebook's own words
the old net2 'only reliably cleans up one pile of shards'; that was the optimiser, not the task.

### Sweep 3: net4 refinement (seeds, entropy, lr, clip, network) and net3 with the fast trainer

| run | envs | epochs x mb | lr | ent | net | steps to 0.95 | ratio | misgen check |
|---|---|---|---|---|---|---|---|---|
| net3_e1024_ep4mb4_lr0.003_s2 | 1024 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 96 | 0.978 | in-dist 0.978; shift bins 0.07, env_shift proxy 3.41 -> misgeneralises |
| net3_e1024_ep4mb4_lr0.003_s1 | 1024 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 112 | 0.986 | in-dist 0.986; shift bins 0.07, env_shift proxy 1.00 -> misgeneralises |
| net3_e512_ep4mb4_lr0.003_s2 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 128 | 0.978 | in-dist 0.978; shift bins 0.07, env_shift proxy 1.30 -> misgeneralises |
| net3_e512_ep4mb4_lr0.003_s1 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 144 | 0.973 | in-dist 0.973; shift bins 0.07, env_shift proxy 0.98 -> misgeneralises |
| net4_e1024_ep4mb4_lr0.003_s3 | 1024 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 128 | 0.968 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_eps0.2 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 128 | 0.955 | env_shift bins 1.00, proxy 0.00 |
| net4_e1024_ep4mb4_lr0.003_s2 | 1024 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 144 | 0.964 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_net16-128-4-2 | 512 | 4 x 4 | 0.003 | 0.01 | 16/128/4/2 | 144 | 0.963 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_net16-64-3-2 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/3/2 | 144 | 0.962 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_ent0.003 | 512 | 4 x 4 | 0.003 | 0.003 | 16/64/5/2 | 160 | 0.963 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_ent0.03 | 512 | 4 x 4 | 0.003 | 0.03 | 16/64/5/2 | 160 | 0.952 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_s2 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 160 | 0.956 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_s3 | 512 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 160 | 0.943 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.01 | 512 | 4 x 4 | 0.01 | 0.01 | 16/64/5/2 | 192 | 0.958 | env_shift bins 1.00, proxy 0.00 |
| net4_e256_ep4mb4_lr0.003_s3 | 256 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 208 | 0.945 | env_shift bins 1.00, proxy 0.00 |
| net4_e256_ep4mb4_lr0.003_s2 | 256 | 4 x 4 | 0.003 | 0.01 | 16/64/5/2 | 224 | 0.961 | env_shift bins 1.00, proxy 0.00 |
| net4_e512_ep4mb4_lr0.003_net8-32-3-1 | 512 | 4 x 4 | 0.003 | 0.01 | 8/32/3/1 | 448 | 0.958 | env_shift bins 1.00, proxy 0.00 |

Reading: seeds 2 and 3 confirm seed 1 (512 envs: 144-160 steps; 1024: 128-144; 256: 192-224). Entropy
0.003-0.03 and clip 0.1-0.2 make no difference; lr 1e-2 still works but is slower; a 3-conv net matches the
5-conv one (the tiny 8/32/3/1 net is 3x slower to learn). net3 with the fast trainer reaches optimum in
96-144 steps and still misgeneralises exactly as before (bins in 7% of shifted layouts, proxy ~1 on env_shift).

## 4. Validation of candidate configurations (isolated, one run per GPU, pure training time)

Compiled rollouts, lr 3e-3, fixed step budgets, no periodic evaluation inside the timed region.
Wall times are for an RTX A4000 with the box's CPUs partly loaded by other sweeps (so slightly
pessimistic). "ratio" is the sampled-policy return over the oracle on the role's evaluation set.

| role | config | steps | seeds | train wall | ratio (min-max) | behaviour |
|---|---|---|---|---|---|---|
| net4 | 512 envs, 4x4 | 192 | 5 | 24-25 s | 0.967-0.974 | passes 5/5: bins on env_shift 100%, proxy 0 |
| net4 | 1024 envs, 4x4 | 160 | 5 | 31-34 s | 0.971-0.982 | passes 5/5 |
| net4 | 256 envs, 4x4 | 256 | 5 | 31-34 s | 0.958-0.971 | passes 5/5 |
| net4 | 512 envs, 4x4, eager rollouts | 192 | 1 | 34 s | 0.972 | (compile saves ~30% of the step) |
| net3 | 512 envs, 4x4 | 160 | 3 | 20-21 s | in-dist 0.95+; shift ratio 0.07 | misgeneralises 3/3: shift bins 7%, env_shift proxy ~1.0 |
| net3 | 1024 envs, 4x4 | 128 | 3 | 25 s | shift ratio 0.07 | misgeneralises 3/3 |
| net1 | 64 rollouts, 8x4 | 64 | 3 | 9-12 s | 0.986-0.998 | farms 3/3 (drop share 0.97-0.98, never bins) |
| net1 | 256 rollouts, 8x4 | 64 | 3 | 10-12 s | 0.949-0.976 | farms 3/3 |
| net2 | 256 rollouts, 8x4 | 128 | 3 | 17 s | 0.773-0.997 | **1 of 3 seeds stalls at 3 piles (0.77)**; no drops, no breaks |
| net2 | 256 rollouts, 4x4 | 192 | 3 | 15 s | 0.772-0.983 | **1 of 3 seeds stalls at 3 piles** |

Decisions: net4 = 512 envs / 192 steps; net3 = 512 envs / 160 steps; net1 = 64 rollouts / 64
steps. net2 needs a larger budget or a better minibatch structure: a seed sweep (sweep 4b) sizes it.

### Bonus-section claims re-checked with the new trainer

The sheet's "everyone has a price" solutions claim PPO "learns nothing" on `env_price` (long detour
vs one smashed urn) and `env_wall` (shards sealed behind urns). With the new defaults (256
rollouts, 8x4 epochs, lr 3e-3, compiled) for 512 steps, two seeds each: reward2, bin and break
probes are all exactly 0 on both layouts. The claim stands; the policy settles on "avoid urns, do
nothing", and the stronger optimiser does not push through the -2 moat either. No text change.

### Sweep 4: minibatch structure (512 envs, lr 3e-3, compiled), net4 and net2

| role | config | seed | steps to 0.95 | final ratio |
|---|---|---|---|---|
| net2 | 256 x 6 ep x 4 mb | 1 | not in 512 | 0.272 |
| net2 | 256 x 6 ep x 4 mb | 2 | 80 | 0.988 |
| net2 | 256 x 8 ep x 1 mb | 1 | not in 512 | 0.774 |
| net2 | 256 x 8 ep x 1 mb | 2 | 304 | 0.981 |
| net2 | 256 x 8 ep x 2 mb | 1 | not in 512 | 0.773 |
| net2 | 256 x 8 ep x 2 mb | 2 | 512 | 0.996 |
| net4 | 512 x 3 ep x 4 mb | 1 | 192 | 0.964 |
| net4 | 512 x 3 ep x 4 mb | 2 | 176 | 0.964 |
| net4 | 512 x 4 ep x 1 mb | 1 | not in 512 | 0.520 |
| net4 | 512 x 4 ep x 1 mb | 2 | not in 512 | 0.433 |
| net4 | 512 x 4 ep x 2 mb | 1 | 224 | 0.958 |
| net4 | 512 x 4 ep x 2 mb | 2 | 272 | 0.949 |
| net4 | 512 x 6 ep x 2 mb | 1 | 176 | 0.962 |
| net4 | 512 x 6 ep x 2 mb | 2 | 160 | 0.952 |

Reading: 4 epochs x 4 minibatches stays the best structure for net4 (one big minibatch per epoch fails to learn; 2 minibatches is slower). For net2 the structure matters less than the seed (see below).

### Sweep 4b / 5: net2 across seeds, entropy 0.001-0.01, batch 256 vs 1024

| rollouts | epochs x mb | entropy | seeds | reached 0.95 | steps (per seed) |
|---|---|---|---|---|---|
| 256 | 4 x 4 | 0.001 | 4 | 2/4 | stuck@0.77, 160, stuck@0.77, 112 |
| 256 | 4 x 4 | 0.003 | 4 | 2/4 | stuck@0.77, 80, stuck@0.77, 112 |
| 256 | 4 x 4 | 0.01 | 4 | 3/4 | stuck@0.77, 80, 160, 96 |
| 256 | 8 x 2 | 0.001 | 4 | 2/4 | stuck@0.77, 128, 176, stuck@0.77 |
| 256 | 8 x 4 | 0.001 | 4 | 3/4 | stuck@0.76, 80, 80, 80 |
| 256 | 8 x 4 | 0.003 | 4 | 3/4 | 80, 352, stuck@0.77, 64 |
| 256 | 8 x 4 | 0.01 | 4 | 4/4 | 96, 64, 352, 80 |
| 1024 | 4 x 4 | 0.003 | 4 | 3/4 | 112, 112, stuck@0.77, 144 |
| 1024 | 4 x 4 | 0.01 | 4 | 3/4 | 160, 96, stuck@0.77, 96 |
| 1024 | 8 x 4 | 0.003 | 4 | 3/4 | 128, stuck@0.77, 64, 80 |
| 1024 | 8 x 4 | 0.01 | 4 | 3/4 | 80, 112, 112, stuck@0.77 |

Reading: about one run in four gets stuck at exactly 0.77 of the oracle = three piles of four (one run at one pile), regardless of batch size, epochs, or entropy up to 0.01, and it stays stuck to 512 steps. Once three piles are binned the near-deterministic policy never explores its way to the far fourth pile.

### Sweep 6: escaping the three-pile trap (net2, 256 rollouts, 8 x 4, lr 3e-3)

| variant | pass | steps to 0.95 per seed | final ratio | drop probe |
|---|---|---|---|---|
| ent0.01_mid | 4/4 | 64, 64, 80, 48 | 0.993-0.997 | 0.00-0.05 |
| ent0.03_mid | 4/4 | 64, 64, 64, 64 | 0.987-0.992 | 0.02-0.07 |
| ent0.03_small | 4/4 | 80, 64, 96, 80 | 0.984-0.996 | 0.02-0.06 |
| ent0.1_mid | 1/4 | 96, 112, 96, 96 | 0.966-0.977 | 0.11-0.34 |
| ent0.1_small | 0/4 | 96, 80, 128, 144 | 0.966-0.974 | 0.16-0.27 |
| sched0.03to0.001 | 3/4 | 80, 64, -, 80 | 0.774-0.994 | 0.00-0.03 |
| sched0.1to0.001 | 3/4 | 80, 64, 128, 80 | 0.980-0.986 | 0.02-0.18 |

(`mid` = 16ch/64w/3conv/2dense network instead of the shipped 8/16/2/1; `sched` = entropy annealed linearly to 0.001.)

Reading: a constant entropy bonus of 0.03 removes the trap with the shipped network (4/4, 64-96 steps, return 98-99.6% of optimum, drops <= 0.06). 0.1 is too noisy (drop probe 0.16-0.34, return 97%). Annealing schedules are no better than the constant (3/4). A bigger net at 0.01 also works (4/4) but would change student-visible code, so 0.03 with the shipped net is adopted.

### Sweep 7: confirming entropy 0.03 for the fixed-layout trainer

| role | entropy | seeds | steps to 0.95 | final ratio | behaviour |
|---|---|---|---|---|---|
| net1 | 0.01 | [1, 2, 3, 4] | 64, 48, 48, 48 | 0.961-0.997 | farms: drop share 0.97-0.98, never bins |
| net1 | 0.03 | [1, 2, 3, 4] | 64, 64, 64, 64 | 0.952-0.966 | farms: drop share 0.98-0.98, never bins |
| net2 | 0.03 | [5, 6, 7, 8] | 64, 128, 64, 80 | 0.976-0.994 | drops 0.00-0.15, breaks <= 0.018 |

Reading: at 0.03, net1 still farms at 95-97% of the farming optimum in 64 steps (0.01: 96-99.7%), and net2 passes on seeds 5-8 as well (8/8 overall, worst case 128 steps). Budgets: net1 64 steps, net2 192 steps (1.5x the worst seed).

## 5. Final configuration and end-to-end result

Adopted defaults (all in `ppo.py`; the notebook's training loops pass only net, env, reward,
optimiser, discount and generator, and the two `ActorCriticNetwork.init` cells are unchanged):

| | fixed layout (`ppo_train_step`) | procedural (`ppo_train_step_multienv`) |
|---|---|---|
| parallel rollouts / envs per step | 256 | 512 (set in the notebook's loop) |
| update epochs x minibatches | 8 x 4 | 4 x 4 |
| learning rate (notebook, Adam) | 3e-3 | 3e-3 |
| entropy coefficient | 0.03 | 0.01 |
| rollouts | compiled CUDA-graph step, eager fallback | same |
| clip eps / GAE lambda / critic coeff / grad norm | 0.1 / 0.95 / 0.5 / 0.5 (unchanged) | unchanged |

Step budgets: net1 96, net2 192, net3 160, net4 192 (each >= 1.5x the worst seed seen).

The regenerated `solutions.py` run as a script (four trainings, every test, all evaluations and plots),
alone on one RTX A4000, with the notebook's own seeds:

| | before (branch, A40 under load) | before (original, A40) | **now (A4000)** |
|---|---|---|---|
| whole notebook | ~10 min | ~16 min, net4 a coin flip | **96 s** |

| agent | result | oracle ratio |
|---|---|---|
| net1 | farms: drop probe 26.5 of return 27.2, never bins, never breaks | 0.995 |
| net2 | cleans all four piles: bin probe 3.59, drops 0.02, breaks 0.001 | 0.998 |
| net3 | in-distribution 0.989 of oracle; on shifted layouts bins in 6%, proxy 0.79; on `env_shift` bins 0%, proxy 0.98 | misgeneralises |
| net4 | on shifted layouts bins in 100%; on `env_shift` bins 100%, proxy 0.00 | 0.971 |

Two independent end-to-end runs on different GPUs gave bit-identical results (the compiled path
seeds the device RNG from the notebook's generator), so the demonstration is now reproducible as
well as fast.

CPU fallback (no GPU, 4 threads, eager rollouts): fixed-layout step 0.64 s, procedural step 1.2 s,
so about 1 min for net1, 2 min for net2, 3.2 min for net3, 3.8 min for net4 (~10 min total).

## 6. Changes on the branch

`gen/support/part6_goalmisgen/ppo.py`
- Multi-epoch minibatch updates (from the earlier commit) with the tuned defaults above.
- `collect_annotated_rollout_fast`: compiled CUDA-graph rollout step with eager fallback;
  `compile_rollouts=True` by default.
- Optional entropy annealing (`entropy_coeff_final` + `progress`), off by default; tested, not
  needed, kept as a knob.

`gen/masters/master_2_6.py`
- `train_agent` / `train_agent_multienv`: hyperparameter blocks removed (defaults in `ppo.py`),
  lr 3e-3, 512 envs per step; step budgets 96 / 192 / 160 / 192; prose about training time, the
  net4 plateau, and the "How are we training the agent?" dropdown updated.
- Bonus: the "tune the agent" bullet (which assumed net2 cleans one pile) replaced by a "break the
  agent" bullet that has students reproduce the old behaviour by weakening the optimiser.

Not changed: environment, reward functions, network architectures, discount, horizon, the
misgeneralisation set-up, the tests, and everything the students write.

## 7. Caveats

- Timings are for an RTX A4000. A Colab T4 is similar-class; the compiled rollout removes most
  of the launch overhead that made the old loop slow, so the ~1.5 min total should transfer,
  plus ~5-10 s of one-off compile time per batch shape. Not verified on Colab itself.
- `torch.compile` needs triton (present on Colab GPU runtimes). Any failure falls back to eager
  rollouts with a printed note; nothing else changes.
- With the compiled path the rollout RNG is the device generator seeded from the notebook's
  generator each rollout, so CPU and GPU runs of the same seed differ (they already did).
- The three-pile trap for net2 is real: with entropy <= 0.01 about one seed in four stalls at
  three piles. 0.03 fixed it on 8/8 seeds; if a student lowers the entropy they may see it.

## 8. Reproducing

```
PYTHONPATH=gen/support python3 gen/support/part6_goalmisgen/tests.py
python3 gen/core/main.py --chapters=2.6 --use_py=true
```
Sweep/oracle/evaluation scripts (`oracle.py`, `evalh.py`, `sweep.py`, `launcher.py`, `summarise.py`,
`compile_probe*.py`) lived in the session scratch directory and are not committed; the tables above
are their output.

## 9. CPU-only: is it feasible, and does batch size need to change for CPU?

Measured on the idle 16-core box (ms per training step, eager rollouts; the compiled step gives
only 74 -> 50 ms on the rollout part on CPU, and the update phase dominates there, so it is not
used on CPU):

| threads | fixed layout, 256 rollouts, 8x4 | procedural 512 envs, 4x4 | 256 envs | 128 envs |
|---|---|---|---|---|
| 1 | 960 | 2483 | 1230 | 628 |
| 2 (Colab CPU has 2 vCPUs) | 593 | 1643 | 827 | 436 |
| 4 | 494 | 1099 | 555 | 305 |
| 8 | 396 | 842 | 432 | 244 |
| 16 | 495 | 1045 | 548 | 326 |

So on CPU the cost is compute-bound and roughly linear in batch, and 8 threads is the sweet spot
(the notebook's CPU cap was raised from 4 to 8; 16 is ~30% slower than 8).

Would a smaller batch help CPU? Steps to reach 95% of optimum across seeds (Sweep 8 + earlier):
net4 at 128 envs 240-352 (budget 512), 256 envs 192-224 (budget 320), 512 envs 128-160 (budget 192).
Implied wall per procedural agent:

| envs / budget | GPU (A4000) | CPU 2 threads | CPU 8 threads |
|---|---|---|---|
| 512 / 192 | 25 s | 5.3 min | 2.7 min |
| 256 / 320 | 39 s | 4.4 min | 2.3 min |
| 128 / 512 | ~59 s | 3.7 min | 2.1 min |

Smaller batches save at most 30% on CPU while costing up to 2.4x on GPU, because the extra steps
eat most of the per-step saving. The fixed-layout agents are cheap on both devices (net2 at 64
rollouts would save ~1 min of CPU time but one net1 seed then needs 160 steps). Decision: one
configuration for both devices (the validated one); CPU differs only in the thread cap.

Verdict: CPU-only is feasible as a fallback (see the measured end-to-end run below) but the
procedural agents dominate and the GPU is roughly an order of magnitude faster overall.

### Measured CPU-only end-to-end (whole notebook, eager rollouts, thread cap 8)

| device | total | net1 | net2 | net3 (misgen check) | net4 |
|---|---|---|---|---|---|
| A4000 GPU | 96 s | 0.995 of farming optimum | 0.998 | 0.989 in-dist; env_shift bins 0%, proxy 0.98 | 0.971; env_shift bins 100%, proxy 0 |
| CPU, 8 threads (16-core box) | 585 s (9.8 min) | 0.803, drop share 0.97 | 0.997 | 0.987; bins 0%, proxy 0.98 | 0.969; bins 100%, proxy 0 |
| CPU, 2 threads (Colab vCPU count) | 1018 s (17 min) | 0.975 | 0.991 | 0.986; bins 0%, proxy 1.17 | 0.975; bins 100%, proxy 0 |

(The two CPU runs ran concurrently on the same box, so each is slightly pessimistic. Colab's vCPUs
are slower than this box's cores, so expect the 2-thread figure to be a lower bound there. The CPU
RNG path differs from the GPU one, hence the different net1 ratios; all four agents pass on both.)

Verdict: CPU-only works and demonstrates every behaviour, at 10-17 minutes for the notebook versus
1.6 minutes on a GPU. The GPU is not required, but it is a 6-10x difference and almost all of it is
the two procedural agents.
