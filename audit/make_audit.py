"""Regenerate audit/master_2_6.annotated.py and audit/ppo.annotated.py from the current files."""
import ast
from pathlib import Path

ROOT = Path("/workspace/HOME/guest/david_quarel/iliad2-aixi")

def block(n, title, body):
    lines = [f"# ===== AUDIT {n}: {title} " + "=" * max(4, 90 - len(title) - len(str(n)))]
    for l in body.strip("\n").splitlines():
        lines.append(("# " + l).rstrip())
    lines.append("# " + "=" * 96)
    return "\n".join(lines) + "\n"

MASTER_NOTES = [
 ("## Setup code", "GPU-runtime note added (student erratum (b))",
  """WHAT: a callout was added under '## Setup code' telling Colab users to select a GPU runtime first.
WHY: a student reported 'nothing goes on the GPU; training is on CPU by default'. Instrumenting the
trainer showed every tensor is on cuda:0 whenever a GPU exists (B, section 'erratum (b)'). What the
student actually hit is that Colab's default runtime has no GPU, and the notebook only mentioned
switching runtime type in section 4, after two training runs. ORIGINAL: no note."""),
 ('if device.type == "cpu" and t.get_num_threads() > 8:', "CPU thread cap (added after device detection)",
  """WHAT: if running on CPU with more than 8 torch threads, cap at 8.
WHY: PyTorch defaults to one thread per core; with this workload (64 sequential env steps per
rollout) the synchronisation overhead outweighs parallelism beyond ~8 threads. Measured with the
final batch sizes on a 16-core box (procedural step, ms): 1 thr 2483, 2: 1643, 4: 1099, 8: 842,
16: 1045; with the old tiny batches a 48-core box was 9x slower at the default than at 4 threads
(B, thread table). No effect on GPU; a no-op on Colab (2 vCPUs). Section 9. ORIGINAL: no cap."""),
 ("<summary>How are we training the agent? (Optional) </summary>", "trainer description updated",
  """WHAT: the optional dropdown now says the trainer does several epochs of minibatch updates per
batch and collects rollouts with a compiled CUDA-graph step, and that hyperparameters have defaults.
ORIGINAL: '(one clipped-surrogate gradient update per batch of rollouts, with GAE advantages; no
minibatch epochs)' - which was accurate, and was the root cause of problems P1 and P2 in audit/README.md."""),
 ("def train_agent(", "train_agent: hyperparameter block removed, lr 1e-3 -> 3e-3, defaults 512/8 -> 128/4",
  """WHAT: the explicit ppo_train_step hyperparameters (num_rollouts=16, num_env_steps=64,
eligibility_rate=0.95, proximity_eps=0.1, critic_coeff=0.5, entropy_coeff=0.001, max_grad_norm=0.5)
were removed from the call; the call now passes net, env, reward_fn, optimiser, discount_rate and
generator only. Adam lr 0.001 -> 0.003. Default num_train_steps 512 -> 128, per_vis 8 -> 4.
WHY: the RL algorithm is a black box for this day, so its settings live as defaults in ppo.py
(ppo_train_step: 256 rollouts, 8 epochs x 4 minibatches, entropy 0.03, compiled rollouts).
Student-facing code got shorter. lr 3e-3 from sweep 2: with epochs, lr 1e-3 leaves net2 at 78% of
optimum (three piles) while 3e-3 reaches 98%+ (section 2, Sweep 2). Values unchanged in effect:
num_env_steps 64, GAE 0.95, clip 0.1, critic 0.5, grad norm 0.5 are still the ppo.py defaults."""),
 ("Training takes well under a minute on a Colab GPU.", "timing sentence",
  """ORIGINAL: 'Training should be relatively quick even with a free Colab GPU.' Now: 'Training takes well
under a minute on a Colab GPU.' Measured: net1 96 steps ~10 s on an A4000 (section 4/5)."""),
 ("net1 = train_agent(", "net1 budget 256 -> 96 steps",
  """WHY: with the new trainer net1 reaches 95-99% of the farming optimum (27.4, exact: walk to the
nearest shard, then alternate PICKUP/PUTDOWN) in 48-64 steps on sweep seeds (Sweeps 2, 7); with the
notebook's seed 42, 64 steps gave 81% and 96 steps gave 99.5% (section 5). The old 256-step run
reached 28% of that optimum (B, replication table)."""),
 ("we recommend `num_train_steps=192` this time", "text 512 -> 192", "Matches the net2 budget below."),
 ("net2 = train_agent(", "net2 budget 512 -> 192 steps",
  """WHY: the old net2 cleaned one pile of four (24% of the optimal return 3.60). With 8x4 epochs, lr
3e-3 and entropy 0.03 it cleans all four (99.8%, section 5). Entropy 0.03 matters: at 0.001-0.01
about one seed in four gets stuck at three piles (Sweeps 4b/5); 0.03 passed 8/8 seeds, worst case
128 steps, so the budget is 1.5x that (Sweeps 6, 7)."""),
 ("# env2 = ...", "bare '#' line removed from the EXERCISE template (P4)",
  """WHAT: ORIGINAL had a line containing only '#' between '# env2 = ...' and '# # YOUR CODE HERE'.
WHY: the generator un-comments an EXERCISE block only if every line starts with '# '; the bare '#'
failed that test, so the student notebook shipped the template fully commented out. Verified fixed
in the regenerated exercises notebook (B, 'Other findings')."""),
 ("def train_agent_multienv(", "train_agent_multienv: 32 -> 512 envs, hyperparameters removed, lr 3e-3, default 512 -> 192 steps",
  """WHAT: envs per step 32 -> 512 (in the earlier commit on this branch: 128 with explicit
num_epochs=4, num_minibatches=4; now those are ppo.py defaults). Explicit hyperparameters removed
(num_env_steps, eligibility_rate, proximity_eps, critic_coeff, entropy_coeff=0.01, max_grad_norm).
Adam lr 0.001 -> 0.003. Default num_train_steps 512 -> 192.
WHY: per-step cost is launch-bound on GPU, so 512 envs cost ~1.3x of 32 envs while giving 16x the
data (section 1 timing table). Sweep 1 (32 configs): steps to 95% of optimum fall from 496 (128
envs, 4x4, lr 1e-3) to 144 (512 envs, 4x4, lr 3e-3); 1024 envs needs fewer steps but costs more
per step; validated on 5 seeds: 512 envs / 192 steps = 24-25 s, all pass (section 4). On CPU a
smaller batch would save at most 30% at up to 2.4x the GPU cost, so one configuration is kept for
both devices (section 9). The entropy 0.01 the original passed explicitly is now the multi-env
default in ppo.py (0.001 was tested and collapses the policy on some seeds, B 'why net4 failed')."""),
 ("many more parallel environments per step, and a slightly longer training run.", "section-4 prose",
  """ORIGINAL: '... a slightly larger policy network, and a longer training time.' and 'Colab users: Now
would be the time to switch over to a GPU or TPU runtime ... Expect the training to otherwise run
pretty slow on CPU only.' 'TPU' removed (the code only detects CUDA/MPS). Timing wording updated:
a minute or two on GPU, many minutes on CPU (section 9)."""),
 ("net3 = train_agent_multienv(", "net3 budget 4096/128 -> 160/8",
  """WHY: with the new trainer net3 reaches 95% of optimum in-distribution in 96-144 steps (Sweep 3)
and still misgeneralises exactly as the section needs: bins in 7% of shifted layouts, 0% on
env_shift, proxy ~1.0 (sections 3-5; 3/3 validation seeds). 160 steps: ~20 s vs ~7 min before."""),
 ("# env_shift = ...", "bare '#' line removed from the EXERCISE template (P4)",
  """Same generator issue as env2, but worse: this cell's HIDE block calls tests.test_env_shift(env_shift)
and tests.test_proxy(proxy), so with the template commented out students got a NameError."""),
 ("# test_env_shift checks only the structural requirements", "leaked bookkeeping comment removed",
  """ORIGINAL had two extra comment lines starting '# #CLAUDE: kept both tests ...' from an earlier
editing session; they were visible in both student notebooks. Removed; the explanatory note stays."""),
 ("## Training out of distribution", "net4 prose: plateau explained, '~5 minute run' claim removed",
  """ORIGINAL: '... we should see goal misgeneralisation decrease (this is another ~5 minute training run):'
WHY: the old 4096-step run failed about half the time (P1); the new text explains the plateau
(return near zero for ~100 steps, then a sharp climb) and tells students what to do if the curve
has not taken off. Evidence: takeoff at step 112-160 across 12+ runs (Sweeps 1, 3, 4)."""),
 ("net4 = train_agent_multienv(", "net4 budget 4096/128 -> 192/8",
  """WHY: see P1 in audit/README.md. Validation, 512 envs / 192 steps, 5 seeds: 24-25 s, 96.7-97.4% of
optimum on 1000 shifted layouts, bins on env_shift 100%, proxy 0.00 (section 4). End-to-end with
the notebook's seed: 97.1% / 100% / 0.00 (section 5)."""),
 ("* **Break the agent.**", "bonus bullet replaced",
  """ORIGINAL: '* **Tune the agent.** Our `reward2` agent only reliably cleans up one pile of shards in
the fixed layout. Play with the architecture, training length, entropy coefficient, and learning
rate to see how much better you can do. Can you get an agent that reliably clears the whole shop?'
WHY: no longer true (net2 clears the shop). The new bullet asks students to weaken the optimiser
(num_epochs=1, num_minibatches=1 and/or lr 0.001) and observe the old one-pile behaviour, which
Sweep 2 shows is exactly what happens."""),
]

PPO_NOTES = [
 ("def ppo_train_step(", "ppo_train_step defaults (fixed single layout)",
  """CHANGED DEFAULTS: num_rollouts 32 -> 256; entropy_coeff 0.001 -> 0.03.
NEW ARGUMENTS: num_epochs (8), num_minibatches (4), compile_rollouts (True), entropy_coeff_final
(None) + progress (None) for an optional anneal. Unchanged: num_env_steps 64, discount 0.995,
eligibility 0.95, clip 0.1, critic 0.5, max_grad_norm 0.5.
WHY: (1) one update per batch is what kept net1 at 28% and net2 at 24% of optimum; 8 epochs of 4
minibatches + lr 3e-3 (set in the notebook's optimiser) gives 99% (Sweep 2). (2) Batch 256: per
step cost is launch-bound on GPU (16 rollouts 68 ms vs 256 rollouts 75 ms eager), so a big batch
is nearly free (section 1); 64 rollouts was tested (Sweeps 8, 9): cheaper per step but one net1
seed then needs 160 steps, so no net gain. (3) Entropy 0.03: with 0.001-0.01 about one seed in four
stops at three of four piles and never explores to the fourth (Sweeps 4b, 5); 0.03 passed 8/8
seeds, 0.1 is too noisy, annealing was no better (Sweeps 6, 7). net1 still farms at 95-97% of
optimum at 0.03 (Sweep 7).
NOTE: the notebook's train_agent passes none of these, so students see a 6-argument call."""),
 ("def ppo_train_step_multienv(", "ppo_train_step_multienv defaults (procedural layouts)",
  """CHANGED DEFAULTS: entropy_coeff 0.001 -> 0.01 (the notebook used to pass 0.01 explicitly with
'# needs more exploration'; 0.001 collapsed the policy to deterministic on some seeds - B, 'why net4
failed'). NEW: num_epochs 4, num_minibatches 4, compile_rollouts True, optional anneal args.
WHY 4x4: Sweep 1 (32 configs) and Sweep 4 (minibatch structure): 4x4 is the best steps-to-target
per unit cost; one minibatch per epoch fails to learn, 2 epochs is clearly worse, 8 epochs costs
more per step for the same steps. The batch size (512) is chosen in the notebook's loop."""),
 ("def _ppo_train_step(", "_ppo_train_step: entropy schedule, compiled-rollout dispatch, multi-epoch minibatch loop",
  """WHAT: (a) if entropy_coeff_final and progress are given, entropy is interpolated linearly (off by
default; tested in Sweep 6b, not adopted, kept as a knob). (b) rollouts come from
collect_annotated_rollout_fast when compile_rollouts, else the original eager function. (c) the
single 'loss -> backward -> step' became: for each epoch, a random permutation of the B*T
transitions split into num_minibatches chunks, one clipped-surrogate update per chunk; metrics are
averaged over the updates.
INVARIANTS: with num_epochs=1, num_minibatches=1 the permutation is skipped and the parameters are
bit-identical to the old single update (verified, B 'Trainer speedup'). With one update the PPO
probability ratio is exactly 1, so the clipping never engaged before (clip fraction 0, KL ~1e-9);
it does now (clip fraction 0.1-0.3 during training).
ORIGINAL: one ppo_loss_fn call on the whole (B, T) batch, one optimiser step."""),
 ("# Fast rollout collection with torch.compile + CUDA graphs", "compiled CUDA-graph rollout step (new)",
  """WHAT: _get_compiled_step builds and caches (per env class / world size / batch size / net /
device) a torch.compile(mode='reduce-overhead') version of one rollout step: observe -> policy ->
multinomial sample -> env.step, returning next state, observation, logits, value and action.
collect_annotated_rollout_fast replays it num_steps times, calling
torch.compiler.cudagraph_mark_step_begin() before each replay and cloning outputs (graph outputs
are overwritten by the next replay), and seeds the device RNG from the caller's generator once per
rollout. On CPU, or on any exception, it falls back to the eager collect_annotated_rollout for the
rest of the process and prints one line.
WHY: rollout collection dominated the train step on GPU (64 sequential steps of tiny kernels).
Measured per 64-step rollout: eager 97 ms (128 envs) / 250 ms (512, shared GPU); compiled 15 / 17
ms (section 3). Full train step at 512 envs 4x4: 247 -> 130 ms. On CPU the update phase dominates
and compiling the step only gives 74 -> 50 ms on the rollout part, so it is not used there.
VERIFIED: compiled env.step bit-identical to eager over 64 random steps; same generator seed ->
identical actions; stored obs/logits/values match env.observe/net; learning progress identical to
eager (ratio 0.920 vs 0.922 after 109 steps); two end-to-end runs bit-identical (sections 3, 5).
PITFALLS AVOIDED: without the step marker the graph is re-recorded every call (5x slower than
eager); a manual whole-rollout CUDAGraph capture fails on the RNG op and poisons the CUDA context.
NOT DONE: compiling the update phase (only 1.35x, and reduce-overhead mode was slower)."""),
 ("def ppo_loss_fn(", "ppo_loss_fn now takes already-flat transitions",
  """WHAT: the (B, T) -> (B*T) flatten moved out to _ppo_train_step so minibatches can be indexed.
The loss itself (clipped surrogate, clipped value loss, entropy term, diagnostics) is unchanged."""),
]

def annotate(src_path, out_path, notes, header, per_cell):
    lines = Path(src_path).read_text().splitlines(keepends=True)
    inserts = []
    for n, (anchor, title, body) in enumerate(notes, 1):
        idx = [i for i, l in enumerate(lines) if anchor in l]
        assert idx, anchor
        i = idx[0]
        if per_cell:
            while i >= 0 and not lines[i].startswith("# ! CELL TYPE:"):
                i -= 1
            assert i >= 0, anchor
        inserts.append((i, block(n, title, body)))
    for i, b in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(i, b)
    out = header + "".join(lines)
    ast.parse(out)
    Path(out_path).write_text(out)
    print(f"{out_path.name}: {len(notes)} audit blocks, parses")

annotate(ROOT / "gen/masters/master_2_6.py", ROOT / "audit/master_2_6.annotated.py", MASTER_NOTES, '''"""
ANNOTATED AUDIT COPY of gen/masters/master_2_6.py (branch goalmisgen-2.6-benchmark vs main @ 7b715a6).

Not used by the notebook generator. Every cell that changed is preceded by an "AUDIT n:" block
outside the cell, quoting the original text, the reason for the change, and the evidence
(section numbers refer to ../goalmisgen-sweep-report.md unless marked B = ../goalmisgen-benchmark-report.md).
Everything else in this file is byte-identical to the current master.
"""
''', True)
annotate(ROOT / "gen/support/part6_goalmisgen/ppo.py", ROOT / "audit/ppo.annotated.py", PPO_NOTES, '''"""
ANNOTATED AUDIT COPY of gen/support/part6_goalmisgen/ppo.py (branch goalmisgen-2.6-benchmark vs main).

Not imported by anything. Each changed function is preceded by an "AUDIT n:" block giving what
changed, why, and the evidence (section numbers: ../goalmisgen-sweep-report.md; B = ../goalmisgen-benchmark-report.md).
Everything else is byte-identical to the current ppo.py. The rest of the file (GAE, the loss
function's maths, the metrics) is unchanged from main.
"""
''', False)
