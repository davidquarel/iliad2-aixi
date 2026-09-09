"""
ANNOTATED AUDIT COPY of gen/support/part6_goalmisgen/ppo.py (branch goalmisgen-2.6-benchmark vs main).

Not imported by anything. Each changed function is preceded by an "AUDIT n:" block giving what
changed, why, and the evidence (section numbers: ../goalmisgen-sweep-report.md; B = ../goalmisgen-benchmark-report.md).
Everything else is byte-identical to the current ppo.py. The rest of the file (GAE, the loss
function's maths, the metrics) is unchanged from main.
"""
"""
Reinforcement learning with (simplified) proximal policy optimisation and
generalised advantage estimation in the pottery shop environment, in batched
PyTorch.

A PyTorch port of the JAX original by Matthew Farrugia-Roberts
(https://github.com/matomatical/reward-lab). Each train step collects a batch
of rollouts with the current policy, estimates advantages with GAE, and then
runs several epochs of clipped-surrogate minibatch updates on that batch.

The defaults were tuned (Sep 2026) so that every agent in the [2.6] notebook
reaches >= 95% of the optimal return in a few hundred steps at most: large
batches of parallel environments (the per-step cost is launch-bound, so a
big batch is almost free), 4-8 update epochs per batch, a 3e-3 learning rate,
and rollouts collected by a torch.compile'd CUDA-graph step when a GPU is
available (see `collect_annotated_rollout_fast`). None of this changes what
is being optimised; it only changes how quickly the optimiser gets there.

Note: in the JAX original, gradient clipping lives inside the optax optimiser
chain; here it is applied explicitly inside the train step (`max_grad_norm`).
Construct the optimiser as a plain `torch.optim.Adam`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from part6_goalmisgen.agent import ActorCriticNetwork
from part6_goalmisgen.evaluation import RewardFunction, compute_return
from part6_goalmisgen.potteryshop import (
    AnnotatedRollout,
    AnnotatedTransition,
    Environment,
    Observation,
    State,
    collect_annotated_rollout,
    tree_map,
)


# ===== AUDIT 1: ppo_train_step defaults (fixed single layout) ============================================
# CHANGED DEFAULTS: num_rollouts 32 -> 256; entropy_coeff 0.001 -> 0.03.
# NEW ARGUMENTS: num_epochs (8), num_minibatches (4), compile_rollouts (True), entropy_coeff_final
# (None) + progress (None) for an optional anneal. Unchanged: num_env_steps 64, discount 0.995,
# eligibility 0.95, clip 0.1, critic 0.5, max_grad_norm 0.5.
# WHY: (1) one update per batch is what kept net1 at 28% and net2 at 24% of optimum; 8 epochs of 4
# minibatches + lr 3e-3 (set in the notebook's optimiser) gives 99% (Sweep 2). (2) Batch 256: per
# step cost is launch-bound on GPU (16 rollouts 68 ms vs 256 rollouts 75 ms eager), so a big batch
# is nearly free (section 1); 64 rollouts was tested (Sweeps 8, 9): cheaper per step but one net1
# seed then needs 160 steps, so no net gain. (3) Entropy 0.03: with 0.001-0.01 about one seed in four
# stops at three of four piles and never explores to the fourth (Sweeps 4b, 5); 0.03 passed 8/8
# seeds, 0.1 is too noisy, annealing was no better (Sweeps 6, 7). net1 still farms at 95-97% of
# optimum at 0.03 (Sweep 7).
# NOTE: the notebook's train_agent passes none of these, so students see a 6-argument call.
# ================================================================================================
def ppo_train_step(
    net: ActorCriticNetwork,
    env: Environment,
    reward_fn: RewardFunction,
    optimiser: torch.optim.Optimizer,
    num_rollouts: int = 256,
    num_env_steps: int = 64,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.03,
    max_grad_norm: float = 0.5,
    num_epochs: int = 8,
    num_minibatches: int = 4,
    compile_rollouts: bool = True,
    entropy_coeff_final: float | None = None,
    progress: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """
    One PPO training step in a single environment: collect `num_rollouts`
    parallel rollouts, then update `net` in place with `num_epochs` passes of
    `num_minibatches` minibatch updates. Returns training metrics.

    The entropy bonus (0.03) is deliberately generous: on the fixed 6x6 layout
    a lower value lets the policy go deterministic after cleaning three piles
    and it never explores its way to the fourth (about one seed in four with
    0.001-0.01); 0.03 fixed that on every seed tried without noticeably
    hurting the return. Pass `progress` (fraction of training done, in [0, 1])
    together with `entropy_coeff_final` to anneal it instead.
    """
    assert env.num_envs is None, (
        "got a batch of environments; use ppo_train_step_multienv"
    )
    return _ppo_train_step(
        net=net,
        env=env,
        reward_fn=reward_fn,
        optimiser=optimiser,
        num_rollouts=num_rollouts,
        num_env_steps=num_env_steps,
        discount_rate=discount_rate,
        eligibility_rate=eligibility_rate,
        proximity_eps=proximity_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        compile_rollouts=compile_rollouts,
        entropy_coeff_final=entropy_coeff_final,
        progress=progress,
        generator=generator,
    )


# ===== AUDIT 2: ppo_train_step_multienv defaults (procedural layouts) ====================================
# CHANGED DEFAULTS: entropy_coeff 0.001 -> 0.01 (the notebook used to pass 0.01 explicitly with
# '# needs more exploration'; 0.001 collapsed the policy to deterministic on some seeds - B, 'why net4
# failed'). NEW: num_epochs 4, num_minibatches 4, compile_rollouts True, optional anneal args.
# WHY 4x4: Sweep 1 (32 configs) and Sweep 4 (minibatch structure): 4x4 is the best steps-to-target
# per unit cost; one minibatch per epoch fails to learn, 2 epochs is clearly worse, 8 epochs costs
# more per step for the same steps. The batch size (512) is chosen in the notebook's loop.
# ================================================================================================
def ppo_train_step_multienv(
    net: ActorCriticNetwork,
    envs: Environment,  # a batch of environments, one per rollout
    reward_fn: RewardFunction,
    optimiser: torch.optim.Optimizer,
    num_env_steps: int = 64,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    num_minibatches: int = 4,
    compile_rollouts: bool = True,
    entropy_coeff_final: float | None = None,
    progress: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """
    One PPO training step across a batch of environments: collect one rollout
    in each environment, then update `net` in place with `num_epochs` passes of
    `num_minibatches` minibatch updates. Returns training metrics.

    The default entropy coefficient is higher than for a single fixed layout
    (0.01 vs 0.001): a distribution of layouts needs more exploration, and a
    lower value can collapse the policy early.
    """
    assert envs.num_envs is not None, (
        "got a single environment; use ppo_train_step (or add a batch "
        "dimension to the environment fields)"
    )
    return _ppo_train_step(
        net=net,
        env=envs,
        reward_fn=reward_fn,
        optimiser=optimiser,
        num_rollouts=None,  # one rollout per environment in the batch
        num_env_steps=num_env_steps,
        discount_rate=discount_rate,
        eligibility_rate=eligibility_rate,
        proximity_eps=proximity_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        compile_rollouts=compile_rollouts,
        entropy_coeff_final=entropy_coeff_final,
        progress=progress,
        generator=generator,
    )


# ===== AUDIT 3: _ppo_train_step: entropy schedule, compiled-rollout dispatch, multi-epoch minibatch loop ====
# WHAT: (a) if entropy_coeff_final and progress are given, entropy is interpolated linearly (off by
# default; tested in Sweep 6b, not adopted, kept as a knob). (b) rollouts come from
# collect_annotated_rollout_fast when compile_rollouts, else the original eager function. (c) the
# single 'loss -> backward -> step' became: for each epoch, a random permutation of the B*T
# transitions split into num_minibatches chunks, one clipped-surrogate update per chunk; metrics are
# averaged over the updates.
# INVARIANTS: with num_epochs=1, num_minibatches=1 the permutation is skipped and the parameters are
# bit-identical to the old single update (verified, B 'Trainer speedup'). With one update the PPO
# probability ratio is exactly 1, so the clipping never engaged before (clip fraction 0, KL ~1e-9);
# it does now (clip fraction 0.1-0.3 during training).
# ORIGINAL: one ppo_loss_fn call on the whole (B, T) batch, one optimiser step.
# ================================================================================================
def _ppo_train_step(
    net: ActorCriticNetwork,
    env: Environment,
    reward_fn: RewardFunction,
    optimiser: torch.optim.Optimizer,
    num_rollouts: int | None,
    num_env_steps: int,
    discount_rate: float,
    eligibility_rate: float,
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
    max_grad_norm: float,
    num_epochs: int,
    num_minibatches: int,
    compile_rollouts: bool,
    entropy_coeff_final: float | None,
    progress: float | None,
    generator: torch.Generator | None,
) -> dict[str, float]:
    # optional entropy schedule: if the caller reports training progress in
    # [0, 1] and an end value, interpolate the entropy coefficient linearly
    # from `entropy_coeff` (start) to `entropy_coeff_final` (end)
    if entropy_coeff_final is not None and progress is not None:
        entropy_coeff = entropy_coeff + (entropy_coeff_final - entropy_coeff) * min(max(progress, 0.0), 1.0)
    # collect experience with current policy...
    if compile_rollouts:
        rollouts = collect_annotated_rollout_fast(env, net, num_env_steps, num_rollouts, generator)
    else:
        rollouts = collect_annotated_rollout(
            env=env,
            policy_value_fn=net.policy_value,
            num_steps=num_env_steps,
            num_rollouts=num_rollouts,
            generator=generator,
        )
    # compute rewards (flatten the batch and time dimensions, apply the
    # reward function to all B*T transitions at once, then reshape)
    B, T = rollouts.transitions.action.shape
    flat_transitions = tree_map(
        lambda x: x.flatten(start_dim=0, end_dim=1),
        rollouts.transitions,
    )
    with torch.no_grad():
        rewards = reward_fn(
            flat_transitions.state,
            flat_transitions.action,
            flat_transitions.next_state,
        ).view(B, T)
    # estimate advantages on the collected experience...
    advantages = generalised_advantage_estimation(
        rewards=rewards,
        values=rollouts.transitions.value_pred,
        final_values=rollouts.final_value_pred,
        eligibility_rate=eligibility_rate,
        discount_rate=discount_rate,
    )
    # update the policy on the collected experience: `num_epochs` passes over
    # the B*T transitions, each split into `num_minibatches` random minibatches.
    # With the defaults (1 epoch, 1 minibatch) this is a single gradient step
    # on the whole batch, in which case the probability ratios are all exactly
    # 1 and the PPO clipping never engages; with more epochs the later updates
    # move away from the rollout policy and the clipping does its job.
    flat_advantages = advantages.flatten()
    num_transitions = flat_advantages.shape[0]
    device = flat_advantages.device
    loss_sum, aux_sums, num_updates = 0.0, {}, 0
    for _ in range(num_epochs):
        # (a single minibatch needs no shuffling; keeping the original order also
        # keeps the default configuration numerically identical to one plain update)
        if num_minibatches == 1:
            perm = torch.arange(num_transitions, device=device)
        else:
            perm = torch.randperm(num_transitions, device=device)
        for indices in perm.chunk(num_minibatches):
            loss, aux = ppo_loss_fn(
                net=net,
                transitions=tree_map(lambda x: x[indices], flat_transitions),
                advantages=flat_advantages[indices],
                proximity_eps=proximity_eps,
                critic_coeff=critic_coeff,
                entropy_coeff=entropy_coeff,
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimiser.step()
            loss_sum += loss.item()
            for k, v in aux.items():
                aux_sums[k] = aux_sums.get(k, 0.0) + v
            num_updates += 1
    # metrics (averaged over the updates)
    train_metrics = {
        "loss": loss_sum / num_updates,
        "return": compute_return(rewards, discount_rate).mean().item(),
        **{k: v / num_updates for k, v in aux_sums.items()},
    }
    return train_metrics


# # #
# ===== AUDIT 4: compiled CUDA-graph rollout step (new) ===================================================
# WHAT: _get_compiled_step builds and caches (per env class / world size / batch size / net /
# device) a torch.compile(mode='reduce-overhead') version of one rollout step: observe -> policy ->
# multinomial sample -> env.step, returning next state, observation, logits, value and action.
# collect_annotated_rollout_fast replays it num_steps times, calling
# torch.compiler.cudagraph_mark_step_begin() before each replay and cloning outputs (graph outputs
# are overwritten by the next replay), and seeds the device RNG from the caller's generator once per
# rollout. On CPU, or on any exception, it falls back to the eager collect_annotated_rollout for the
# rest of the process and prints one line.
# WHY: rollout collection dominated the train step on GPU (64 sequential steps of tiny kernels).
# Measured per 64-step rollout: eager 97 ms (128 envs) / 250 ms (512, shared GPU); compiled 15 / 17
# ms (section 3). Full train step at 512 envs 4x4: 247 -> 130 ms. On CPU the update phase dominates
# and compiling the step only gives 74 -> 50 ms on the rollout part, so it is not used there.
# VERIFIED: compiled env.step bit-identical to eager over 64 random steps; same generator seed ->
# identical actions; stored obs/logits/values match env.observe/net; learning progress identical to
# eager (ratio 0.920 vs 0.922 after 109 steps); two end-to-end runs bit-identical (sections 3, 5).
# PITFALLS AVOIDED: without the step marker the graph is re-recorded every call (5x slower than
# eager); a manual whole-rollout CUDAGraph capture fails on the RNG op and poisons the CUDA context.
# NOT DONE: compiling the update phase (only 1.35x, and reduce-overhead mode was slower).
# ================================================================================================
# Fast rollout collection with torch.compile + CUDA graphs
#
# Collecting rollouts is the dominant cost of a training step: 64 sequential
# environment steps, each a few dozen tiny kernels plus Python overhead, so
# the GPU sits idle most of the time. Compiling one (observe -> policy ->
# sample -> step) function with mode="reduce-overhead" replays it as a CUDA
# graph and cuts the per-rollout cost several-fold. Semantics are unchanged:
# the compiled environment step is bit-identical to `Environment.step`; only
# the action sampling uses the global CUDA RNG (seeded from `generator` at the
# start of each rollout) instead of a CPU generator.
#
# Anything unusual (CPU device, missing triton, an old torch) makes us fall
# back to the plain `collect_annotated_rollout`, so this is purely a speed-up.

_COMPILED_STEPS: dict = {}
_COMPILE_DISABLED = False


def _get_compiled_step(env: Environment, net: ActorCriticNetwork, batch_size: int):
    key = (id(env.__class__), env.world_size, batch_size, id(net), str(env.device))
    fn = _COMPILED_STEPS.get(key)
    if fn is None:
        ws = env.world_size

        def one_step(robot_pos, bin_pos, items_map, inventory):
            state = State(robot_pos=robot_pos, bin_pos=bin_pos, items_map=items_map, inventory=inventory)
            obs = env.observe(state)
            action_logits, value_pred = net(obs)
            action = torch.multinomial(torch.softmax(action_logits, dim=-1), 1).squeeze(-1)
            nxt = env.step(state, action)
            return (nxt.robot_pos, nxt.bin_pos, nxt.items_map, nxt.inventory,
                    obs.grid, obs.vec, action_logits, value_pred, action)

        fn = torch.compile(one_step, mode="reduce-overhead", dynamic=False)
        _COMPILED_STEPS[key] = fn
    return fn


@torch.no_grad()
def collect_annotated_rollout_fast(
    env: Environment,
    net: ActorCriticNetwork,
    num_steps: int,
    num_rollouts: int | None = None,
    generator: torch.Generator | None = None,
) -> AnnotatedRollout:
    """
    Same contract as `collect_annotated_rollout(env, net.policy_value, ...)`,
    using a compiled, CUDA-graph-replayed per-step function on the GPU. Falls
    back to the eager implementation on CPU or if compilation fails.
    """
    global _COMPILE_DISABLED
    if _COMPILE_DISABLED or env.device.type != "cuda":
        return collect_annotated_rollout(env, net.policy_value, num_steps, num_rollouts, generator)
    try:
        state = env.reset(num_rollouts)
        B = state.inventory.shape[0]
        step_fn = _get_compiled_step(env, net, B)
        if generator is not None:  # derive this rollout's device RNG seed from the caller's generator
            torch.cuda.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
        args = (state.robot_pos, state.bin_pos, state.items_map, state.inventory)
        transitions = []
        for _ in range(num_steps):
            torch.compiler.cudagraph_mark_step_begin()
            out = step_fn(*args)
            out = [x.clone() for x in out]  # graph outputs are overwritten by the next replay
            robot_pos, bin_pos, items_map, inventory, grid, vec, action_logits, value_pred, action = out
            next_state = State(robot_pos=robot_pos, bin_pos=bin_pos, items_map=items_map, inventory=inventory)
            transitions.append(
                AnnotatedTransition(
                    state=state,
                    obs=Observation(grid=grid, vec=vec),
                    value_pred=value_pred,
                    action=action,
                    action_logits=action_logits,
                    next_state=next_state,
                )
            )
            state = next_state
            args = (robot_pos, bin_pos, items_map, inventory)
        final_obs = env.observe(state)
        _, final_value_pred = net.policy_value(final_obs)
    except Exception as e:  # noqa: BLE001 - any compile/runtime problem => eager path from now on
        _COMPILE_DISABLED = True
        print(f"[ppo] compiled rollouts unavailable ({type(e).__name__}: {str(e)[:80]}); using eager rollouts")
        return collect_annotated_rollout(env, net.policy_value, num_steps, num_rollouts, generator)
    stacked = tree_map(lambda *xs: torch.stack(xs, dim=1), transitions[0], *transitions[1:])
    return AnnotatedRollout(transitions=stacked, final_obs=final_obs, final_value_pred=final_value_pred)


# # #
# PPO loss function


# ===== AUDIT 5: ppo_loss_fn now takes already-flat transitions ===========================================
# WHAT: the (B, T) -> (B*T) flatten moved out to _ppo_train_step so minibatches can be indexed.
# The loss itself (clipped surrogate, clipped value loss, entropy term, diagnostics) is unchanged.
# ================================================================================================
def ppo_loss_fn(
    net: ActorCriticNetwork,
    transitions: AnnotatedTransition,  # one leading (flat) batch dimension
    advantages: Float[Tensor, "batch_size"],
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    batch_size = advantages.shape[0]
    batch = torch.arange(batch_size, device=advantages.device)

    # run network to get latest predictions
    new_action_logits, new_value_preds = net.policy_value(transitions.obs)
    # -> float[batch_size, 7], float[batch_size]

    # actor loss
    new_action_logprobs = F.log_softmax(new_action_logits, dim=1)
    new_chosen_logprobs = new_action_logprobs[batch, transitions.action]
    old_action_logprobs = F.log_softmax(transitions.action_logits, dim=1)
    old_chosen_logprobs = old_action_logprobs[batch, transitions.action]
    action_log_ratios = new_chosen_logprobs - old_chosen_logprobs
    action_prob_ratios = torch.exp(action_log_ratios)
    action_prob_ratios_clipped = torch.clamp(
        action_prob_ratios,
        1 - proximity_eps,
        1 + proximity_eps,
    )
    std_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -torch.minimum(
        std_advantages * action_prob_ratios,
        std_advantages * action_prob_ratios_clipped,
    ).mean()

    # critic loss
    value_diffs = new_value_preds - transitions.value_pred
    value_diffs_clipped = torch.clamp(
        value_diffs,
        -proximity_eps,
        proximity_eps,
    )
    new_value_preds_proximal = transitions.value_pred + value_diffs_clipped
    targets = transitions.value_pred + advantages
    critic_loss = (
        torch.maximum(
            torch.square(new_value_preds - targets),
            torch.square(new_value_preds_proximal - targets),
        ).mean()
        / 2
    )

    # entropy regularisation term
    per_step_entropy = -torch.sum(
        torch.exp(new_action_logprobs) * new_action_logprobs,
        dim=1,
    )
    average_entropy = per_step_entropy.mean()

    # diagnostics
    with torch.no_grad():
        actor_clipfrac = (action_prob_ratios_clipped != action_prob_ratios).float().mean()
        actor_approxkl1 = (-action_log_ratios).mean()
        actor_approxkl3 = ((action_prob_ratios - 1) - action_log_ratios).mean()
        critic_clipfrac = (value_diffs != value_diffs_clipped).float().mean()

    # total loss
    total_loss = (
        actor_loss + critic_coeff * critic_loss - entropy_coeff * average_entropy
    )
    return (
        total_loss,
        {
            "loss-actor": actor_loss.item(),
            "loss-critic": critic_loss.item(),
            "entropy": average_entropy.item(),
            "actor-clip": actor_clipfrac.item(),
            "critic-clip": critic_clipfrac.item(),
            "actor-kl1": actor_approxkl1.item(),
            "actor-kl3": actor_approxkl3.item(),
        },
    )


# # #
# Generalised advantage estimation


def generalised_advantage_estimation(
    rewards: Float[Tensor, "B num_steps"],
    values: Float[Tensor, "B num_steps"],
    final_values: Float[Tensor, "B"],
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Tensor, "B num_steps"]:
    """
    Compute GAE advantages for a batch of rollouts with a reverse scan
    through the time axis.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    next_values = final_values
    for t in reversed(range(T)):
        gae = (
            rewards[:, t]
            - values[:, t]
            + discount_rate * (next_values + eligibility_rate * gae)
        )
        advantages[:, t] = gae
        next_values = values[:, t]
    return advantages
