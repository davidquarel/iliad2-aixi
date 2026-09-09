"""
Reinforcement learning with (simplified) proximal policy optimisation and
generalised advantage estimation in the pottery shop environment, in batched
PyTorch.

A PyTorch port of the JAX original by Matthew Farrugia-Roberts
(https://github.com/matomatical/reward-lab). Each train step collects a batch
of rollouts with the current policy, estimates advantages with GAE, and
performs a single clipped-surrogate gradient update (no minibatch epochs).

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
    AnnotatedTransition,
    Environment,
    collect_annotated_rollout,
    tree_map,
)


def ppo_train_step(
    net: ActorCriticNetwork,
    env: Environment,
    reward_fn: RewardFunction,
    optimiser: torch.optim.Optimizer,
    num_rollouts: int = 32,
    num_env_steps: int = 64,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
    max_grad_norm: float = 0.5,
    num_epochs: int = 1,
    num_minibatches: int = 1,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """
    One PPO training step in a single environment: collect `num_rollouts`
    parallel rollouts, then update `net` in place. Returns training metrics.
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
        generator=generator,
    )


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
    entropy_coeff: float = 0.001,
    max_grad_norm: float = 0.5,
    num_epochs: int = 1,
    num_minibatches: int = 1,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """
    One PPO training step across a batch of environments: collect one rollout
    in each environment, then update `net` in place. Returns training metrics.
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
        generator=generator,
    )


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
    generator: torch.Generator | None,
) -> dict[str, float]:
    # collect experience with current policy...
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
# PPO loss function


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
