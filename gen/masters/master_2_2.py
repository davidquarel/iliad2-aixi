# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
{
    "sections": [
        {"title": "VPG", "icon": "1-circle-fill", "subtitle": "(100%)"},
    ],
}
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [2.2] - Vanilla Policy Gradient (VPG)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
In this notebook you'll implement **Vanilla Policy Gradient (VPG)**, the first policy gradient algorithm and the foundation that many modern RL algorithms (including PPO) build on. You'll apply it to the classic CartPole environment.

(This material is adapted from the second half of the ARENA "DQN & VPG" day; the DQN half is omitted here.)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ VPG

The Policy Gradient Theorem is what all policy gradient methods are based on: it lets us compute the gradient of the expected return, something that would naively not have a well-defined gradient. We'll then implement Vanilla Policy Gradient (VPG) on the CartPole environment.

> ##### Learning Objectives
>
> - Understand the Policy Gradient Theorem
> - Understand the VPG algorithm: how to perform on-policy policy gradient
> - Implement VPG using PyTorch, on the CartPole environment
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup (don't read, just run!)
'''

# ! CELL TYPE: code
# ! FILTERS: [~]
# ! TAGS: []

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: [master-comment]

# # ILIAD Intensive setup. On Colab: install deps and pull this notebook's support
# # modules (gpu_env, plotly_utils, rl_utils, part1_intro_to_rl/, part2_.../) from the
# # auto-built `notebooks` branch. opencv-python + imageio-ffmpeg power the cartpole video.
# import os
# import sys

# if "google.colab" in sys.modules:
#     %pip install -q torch numpy "gymnasium==0.29.0" jaxtyping tqdm "plotly>=5" opencv-python pandas imageio imageio-ffmpeg torchinfo wandb
#     !git clone --depth 1 -b notebooks --filter=blob:none --sparse https://github.com/davidquarel/iliad2-aixi /content/iliad
#     !cd /content/iliad && git sparse-checkout set part_vpg
#     if "/content/iliad/part_vpg" not in sys.path:
#         sys.path.insert(0, "/content/iliad/part_vpg")  # so gpu_env / part2_.../ etc. resolve
#     os.chdir("/content/iliad/part_vpg")
#
# # Run W&B in offline mode so training works without a login; set to "online" if you have an account.
# os.environ.setdefault("WANDB_MODE", "offline")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import os
import sys
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch as t
import torch.nn.functional as F
import wandb
from gymnasium.spaces import Box, Discrete
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torchinfo import summary
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_MODE", "offline")  # run without a W&B login (set "online" if you have one)

Arr = np.ndarray
ActType = Int
ObsType = Int

# Make sure exercises are in the path (on Colab this is handled by the setup cell above)
# FILTERS: ~colab
chapter = "chapter2_rl"
section = "part2_q_learning_and_policy_gradient"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
# END FILTERS

from gpu_env import CartPole
import part2_q_learning_and_policy_gradient.tests as tests
import part2_q_learning_and_policy_gradient.utils as utils
from part1_intro_to_rl.utils import set_global_seeds
from part2_q_learning_and_policy_gradient.probe import Probe4, Probe5
from part2_q_learning_and_policy_gradient.utils import make_env
from plotly_utils import line, plot_cartpole_obs_and_dones
from rl_utils import generate_and_plot_trajectory, make_env

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ VPG
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Policy Gradient Theorem

Instead of learning action-values and deriving a policy (as in Q-learning or DQN), **policy gradient methods learn the policy directly**.
- Policy is parameterized: $\pi_\theta(a|s)$ with parameters $\theta$ (often a neural network).  
- Objective: Choose $\theta$ to maximize expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$ (joy), where $\tau$ is a trajectory and $G(\tau)$ its return.  

We would desire to update the policy directly via **gradient ascent** against $J(\theta)$:
$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

The problem is that the return is a sum of rewards from the trajectory, and the trajectory itself is a result of sampling from the policy, over and over, 
as well as being dependant on the environmental distribution, which we do not have access to.
There is no clear way to directly compute the gradient of the return with respect to the policy parameters.
The solution here is the **policy gradient theorem**, which states that we can instead use the return weighted by the gradient of the log-probability as an unbiased estimator of the gradient of the return.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Derivation</summary>

The probability of sampling a trajectory 
$\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ 
is given by
$$
\Pr(\tau|\theta) = \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)\,\mu(s_{t+1}|s_t, a_t)
$$
where $\mu$ is the environment transition probability.

$$
\begin{align*}
   \nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)] \\
   &= \nabla_\theta \sum_\tau  \Pr(\tau|\theta) \, G(\tau) \\
   &= \sum_\tau \nabla_\theta \Pr(\tau|\theta) \, G(\tau) \\
   &= \sum_\tau \Pr(\tau|\theta)\,\nabla_\theta \log \Pr(\tau|\theta) \, G(\tau) \\
   &= \mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log \Pr(\tau|\theta) \, G(\tau) \right]
   \end{align*}
   $$
   where we made use of the log-derivative trick: $\nabla_\theta p(x) = p(x) \nabla_\theta \log p(x)$.
  
 
   The dynamics $\mu$ do not depend on $\theta$, so:
   $$
   \begin{align*}
   \log \Pr(\tau|\theta) &= \log \left( \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)\,\mu(s_{t+1}|s_t, a_t) \right) \\
   &= \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log \mu(s_{t+1}|s_t, a_t) \\
   &= \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \text{const.}
   \end{align*}
   $$
   where the const. term is independent of $\theta$, so when we take the gradient, it vanishes.

   Thus:
   $$
   \nabla_\theta \log \Pr(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
   $$

Plugging back into the gradient:
$$
\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim \pi_\theta} \left[
  \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\, G(\tau)
\right]
$$

This is the **Vanilla Policy Gradient estimator**, also called **REINFORCE**. 
Each $\log \pi_\theta(a_t|s_t)$ is multiplied by the **full return** $G(\tau)$. 
However, the action $a_t$ cannot influence rewards before time $t$, only those afterwards.
This means that all the rewards before timestep $t$ merely add noise, as no changes to the policy
can affect them.  To reduce variance, replace $G(\tau)$ with the return $G_t$ at timestep $t$, 
also called the **reward-to-go**:
$$
G_t = \sum_{i=t}^{T} \gamma^{i-t} r_{i}
$$

Thus, the lower-variance unbiased estimator is:
$$
\nabla_\theta J(\theta) =
\mathbb{E}_{\tau \sim \pi_\theta} \left[
  \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t
\right]
$$

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
There are many other variants of the policy gradient estimator, as described in [Schulman, 2018](https://arxiv.org/abs/1506.02438).

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/img/policy_grad.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Implementation
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We make use of the same CartPole environment as before, but now we have a vectorized version that is entirely defined in terms of tensor operations (see `chapter2_rl/exercises/gpu_env.py`). This environment is identical to the one used for DQN, but it now runs entirely on the GPU. This means
* we don't need to constantly convert between numpy and torch tensors
* we can run large numbers of environments in parallel (~thousands of environments for ~millions of environmental steps per second)
* we avoid copying data back and forth between the CPU and GPU, which can be a significant bottleneck
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Policy Network

Here, the policy is learned directly as a neural network, rather than learning a Q-value table approximator. We'll use the same architecture as the Q-network from DQN, so we've just included that here for you.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class PolicyNetwork(nn.Module):
    """
    For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.
    """

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        # assert len(obs_shape) == 1, f"Expecting a single vector of observations, got {obs_shape}"
        assert len(hidden_sizes) == 2, f"Expecting 2 hidden layers, got {len(hidden_sizes)}"
        self.layers = nn.Sequential(
            nn.Linear(obs_shape[-1], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


net = PolicyNetwork(obs_shape=(4,), num_actions=2)
summary(net)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Rollout Buffer

The way that our implementation of VPG will work is simple: we perform a rollout across `num_envs` many environments in parallel, and store the trajectories for each. We then learn from that set of rollouts, and then discard it afterwards. One rollout, one learning step. This means we are always learning **on-policy**: we only every learn from data that the current model actually generated. We will use a rollout buffer to store the trajectories.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `Rollout Buffer`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 20 minutes on this exercise.
> ```

The `Rollout` class will store a set of `num_envs` many trajectories. We do not shuffle up anything, or break up a episode into little experiences as we did for DQN. The smallest datapoint is one full trajectory:

$$\tau = s_0 \; a_0 \; r_0 \; s_1 \; a_1 \; r_1 \ldots s_T \; a_T \; r_T$$

The following methods need to be completed:

* `add_step` - adds information gathered from timestep $t$ to the rollout buffer
* `get_batches` - returns a list of `RolloutTensors` objects, each containing `batch_size` many trajectories.

We store the tensors for each step as seperate lists, and then stack once at the end to get the final tensors with the `.get` function. This ends up being cheaper as it avoids spinning up indexed-write kernels per step.

<details>
<summary>Hint</summary>

Use `t.split` to write `get_batches`.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

RolloutTensors = namedtuple("RolloutTensors", ["obs", "actions", "logprobs", "rewards", "dones"])


class Rollout:
    _obs: list[Float[Tensor, " max_steps *obs_shape"]]
    _actions: list[Int[Tensor, " max_steps *action_shape"]]
    _logprobs: list[Float[Tensor, " max_steps"]]
    _rewards: list[Float[Tensor, " max_steps"]]
    _dones: list[Bool[Tensor, " max_steps"]]
    timestep: int

    def __init__(
        self, num_envs: int, max_steps: int, obs_shape: tuple[int], action_shape: tuple[int], device: t.device
    ):
        """
        Args:
            num_envs: number of environments to rollout
            max_steps: maximum number of steps to rollout per environment
            obs_shape: shape of the observation
            action_shape: shape of the action
            device: device to use
        """

        self.MAX_SIZE = max_steps
      
        # Per-step we append tensor references to Python lists (free) and t.stack() once at the
        # end, instead of 5 indexed-write kernels per step into a preallocated buffer. Each stored
        # tensor is freshly produced per step (the env returns a new state tensor each step), so
        # holding references is safe. This removes ~2500 tiny kernel launches per full rollout.
        self._obs, self._actions, self._logprobs, self._rewards, self._dones = [], [], [], [], []
        self.timestep = 0

    def add_step(
        self,
        obs: Float[Tensor, " num_envs *obs_shape"],
        actions: Int[Tensor, " num_envs *action_shape"],
        logprobs: Float[Tensor, " num_envs"],
        rewards: Float[Tensor, " num_envs"],
        dones: Bool[Tensor, " num_envs"],
        infos: dict[str, Any],
    ):
        """
        Adds information to the replay buffer for the current self.timestep
        Don't forget to increment self.timestep afterwards!
        """

        if self.timestep >= self.MAX_SIZE:
            raise ValueError("Rollout is full, cannot add more steps")

        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        # `infos` is intentionally not stored: it is never read during training, and the env
        # builds a fresh GPU clone in it every step (memory + overhead for nothing).
        self._obs.append(obs)
        self._actions.append(actions)
        self._logprobs.append(logprobs)
        self._rewards.append(rewards)
        self._dones.append(dones)
        self.timestep += 1
        # END SOLUTION

    def reset(self):
        self._obs.clear(); self._actions.clear(); self._logprobs.clear()
        self._rewards.clear(); self._dones.clear()
        self.timestep = 0

    def get(self) -> tuple[Tensor, ...]:
        """
        Stack the per-step lists into (num_envs, timestep, ...) tensors. Rollouts can stop early
        (see gen_rollout), so the time dimension is however many steps were actually collected.
        """
        assert self.timestep > 0, "Rollout is empty"
        return RolloutTensors(
            t.stack(self._obs, dim=1),
            t.stack(self._actions, dim=1),
            t.stack(self._logprobs, dim=1),
            t.stack(self._rewards, dim=1).float(),
            t.stack(self._dones, dim=1),
        )

    def get_batches(self, batch_size: int) -> list[RolloutTensors]:
        """
        Splits the rollout buffer into batches of size `batch_size`, and returns a list of
        `RolloutTensors` objects, each containing `batch_size` many trajectories.
        """

        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        tau = self.get()  # filled portion only
        obs = t.split(tau.obs, batch_size, dim=0)
        acts = t.split(tau.actions, batch_size, dim=0)
        logprobs = t.split(tau.logprobs, batch_size, dim=0)
        rewards = t.split(tau.rewards, batch_size, dim=0)
        dones = t.split(tau.dones, batch_size, dim=0)

        batches = [RolloutTensors(*tensors) for tensors in zip(obs, acts, logprobs, rewards, dones)]

        return batches
        # END SOLUTION


tests.test_rollout(Rollout)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## VPG Args

We've provided a dataclass for the training arguments, and will explain as needed later on.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass
class VPGArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-gpu"

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "VPGCartPole"
    wandb_entity: str | None = None
    video_log_freq: int | None = 50   # every N rollouts, render a 4x4 grid video of the rollout
                                      # (logged to wandb if use_wandb, shown inline if live_viz)

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 500_000
    # max_rollout_steps: int = 500
    # min_rollout_steps: int = 64
    num_envs: int = 4

    num_steps_per_rollout: int = 128

    lr: float = 2.5e-4
    gamma: float = 1
    frac_dead_rollout: float = 1
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    rollout_use_count: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    compile: bool = False
    device: str = "cpu"
    normalize_returns: bool = True
    show_probs: bool = False
    num_batches_per_rollout: int = 1
    # LR decay settings
    use_lr_decay: bool = False
    lr_end: Optional[float] = None
    lr_frac: Optional[float] = None
    use_iw: bool = False
    early_stop: bool = True   # cut a rollout short once every env has died at least once
    full_reset: bool = True   # fully reset all envs at the start of each rollout
    live_viz: bool = False    # also display the logged grid video inline (notebook only)

    def __post_init__(self):
        self.batch_size = self.num_envs // self.num_batches_per_rollout
        self.device = t.device(self.device)

        if self.use_lr_decay:
            assert self.lr_end is not None, "lr_end must be set if use_lr_decay is True"
            assert self.lr_frac is not None, "lr_frac must be set if use_lr_decay is True"

        self.env_steps_per_update = self.num_steps_per_rollout * self.num_envs // self.num_batches_per_rollout

        if not self.use_iw:
            assert self.rollout_use_count == 1, "rollout_use_count must be 1 if use_iw is False"
            assert self.num_batches_per_rollout == 1, "num_batches_per_rollout must be 1 if use_iw is False"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## VPG Agent

The following class will be our agent, that will generate rollouts via interaction between the agent and environment, as well as generate actions my sampling them from the policy network. Recall that the policy network now maps observations to logits for each action, so we can sample actions from the distribution.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `VPGAgent`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement the functions:
* `gen_rollout` - this function computes the episode rollout, by interacting with the environment for `args.num_steps_per_rollout` steps. If an episode terminates, we reset the environment and continue. We will track the length of the episode in the `lifespan` variable, which indicates how long each episode runs before termination. For the cartpole environment, this will allow us to track performance (the longer the cart lives, the better it does.)

* `get_actions` - this function takes in an observation, and returns the actions, logprobs, and entropy for that observation. You can use `t.distributions.Categorical(logits=logits)` to construct a distribution, from which you can get the actions, logprobs, and entropy. [See the docs](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical) for details.

Internally, this function also tracks `dead` and `lifespan`, which are tensors of shape `(num_envs,)` that indicate whether each environment is dead and how long each environment has survived respectively.
This will be useful for displaying later on during training so we can get an idea of how long each rollout lasts for.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class VPGAgent:
    """Base Agent class handling the interaction with the environment."""

    dead: Bool[Tensor, " num_envs"]
    lifespan: Int[Tensor, " num_envs"]

    def __init__(
        self,
        envs: gym.Env,
        policy_network: PolicyNetwork,
        args: VPGArgs,
        rng: Optional[np.random.Generator] = None,
    ):
        self.envs = envs
        self.policy_network = policy_network
        self.rng = rng
        self.args = args
        self.obs_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.shape

    @t.no_grad()
    def gen_rollout(self, rollout: Rollout) -> tuple[Rollout, dict[str, Any]]:
        """
        Compute the full episode rollout for all environments in parallel, adding them to the rollout buffer.
        It then returns the rollout buffer, and a dictionary of info contining the lifespan.

        Returns `infos` (list of dictionaries containing info we will log).
        """
        device = self.args.device

        # Force a *full* reset so every env starts a fresh episode aligned to this rollout's window
        # (otherwise the env's internal timestep persists across rollouts and survivors truncate
        # mid-rollout). CartPole resets every env whose terminated|truncated flag is set.
        if self.args.full_reset and hasattr(self.envs, "_terminated"):
            # `terminated`/`truncated` are methods; the settable per-env flags reset() reads are
            # the `_terminated`/`_truncated` tensors. Force them so reset() resets every env.
            self.envs._terminated[:] = True
            self.envs._truncated[:] = True
        obs, _ = self.envs.reset()  # Need a starting observation

        dead = t.zeros(self.args.num_envs, dtype=t.bool, device=device)
        lifespan = t.zeros(self.args.num_envs, dtype=t.int32, device=device)
        rollout.reset()
        early_stop = self.args.early_stop

        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        for timestep in range(self.args.num_steps_per_rollout):
            actions, logprobs, entropy = self.get_actions(obs)
            new_obs, rewards, terminates, truncates, info = self.envs.step(actions)
            # Mask returns at episode boundaries on EITHER termination or truncation: the env
            # auto-resets on both, so returns must not bootstrap across the reset.
            done = terminates | truncates
            rollout.add_step(obs, actions, logprobs, rewards, done, info)
            obs = new_obs
            # Lifespan / convergence are about surviving (not terminating); truncation = success.
            dead = dead | terminates
            lifespan += ~dead
            # Early stop: once every env has died at least once there is no more survival signal to
            # gather this rollout, so cut it short. Near convergence survivors keep dead=False, so
            # we still run the full num_steps_per_rollout (preserving the "survive 500" check).
            if early_stop and (timestep % 16) == 15 and bool(dead.all()):
                break
        # END SOLUTION

        info = {"lifespan": lifespan}

        return rollout, info

    def get_actions(
        self, obs: Float[Tensor, " num_envs *obs_shape"]
    ) -> tuple[Int[Tensor, " num_envs *action_shape"], Float[Tensor, " num_envs"], Float[Tensor, " num_envs"]]:
        """
        Computes the agents turn: given an observation for each environment,
        sample the action the agent takes, along with the log_probs of that action,
        and the entropy of the action distribution.
        Use t.multinomial to sample the actions.
        """
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        # Manual sampling instead of t.distributions.Categorical: this is the hot loop (run
        # num_steps_per_rollout times per rollout), and Categorical's object creation adds large
        # Python overhead per step. Entropy is unused here (it is recomputed in
        # compute_logprobs_and_entropy for the loss), so we skip it entirely.
        logits = self.policy_network(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        actions = t.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        # log pi(a | s) for the sampled action, per env, by advanced indexing (no gather):
        logprobs = log_probs[t.arange(actions.shape[0], device=log_probs.device), actions]
        return actions, logprobs, None
        # END SOLUTION


tests.test_get_actions(VPGAgent, PolicyNetwork)
tests.test_gen_rollout(VPGAgent, PolicyNetwork, VPGArgs, Rollout)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Returns

To compute the REINFORCE loss, we need to compute the return for each step in the trajectory. This gets a little messy as trajectories may be of different lengths, so an episode may have terminated part way through the rollout. You'll need to walk backward through the trajectory, and compute the return for each step.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `compute_returns`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

Compute the returns for each trajectory. Easiest to write as a simple reverse for-loop for now, though if you wish later on you can try a vectorized solution.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compute_returns(
    rewards: Float[Tensor, " num_envs num_steps"], done: Bool[Tensor, " num_envs num_steps"], gamma: float = 0.9
):
    """
    ARGS:
        rewards: The rewards for each trajectory
        done: A boolean tensor indicating if an episode finished on the current timestep
        gamma: The discount factor

    Returns:
        The returns G_t for each trajectory.

        For example:
        - If Rewards = [0, 0, 1, 0, 1]
        - And Done   = [0, 0, 1, 0, 1]
        - Then Returns = [g**2, g, 1, g, 1]
    """
    num_envs, num_steps = rewards.shape

    returns = t.zeros_like(rewards)

    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION

    G = t.zeros_like(rewards[:, 0])  # (num_envs)
    for i in reversed(range(num_steps)):
        G = rewards[:, i] + gamma * G * (~done[:, i])
        returns[:, i] = G
    return returns
    # END SOLUTION


tests.test_compute_returns(compute_returns)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `compute_logprobs_and_entropy`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

Computes the logprobs of actions taken, and the entropy of the action distribution on each timestep. Needed for the loss function.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compute_logprobs_and_entropy(
    tau: RolloutTensors, pi: PolicyNetwork
) -> tuple[Float[Tensor, " num_envs num_steps"], Float[Tensor, " num_envs num_steps"]]:
    """
    Computes the logprobs and entropy of the action distribution on each timestep.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logits = pi(tau.obs)
    log_probs = F.log_softmax(logits, dim=-1)
    # pick out log pi(a_t | s_t) per (env, step) by advanced indexing: index dims 0,1 with
    # broadcast aranges and dim 2 (actions) with the taken action -> shape (num_envs, num_steps).
    num_envs, num_steps = tau.actions.shape
    envs = t.arange(num_envs, device=log_probs.device)[:, None]
    steps = t.arange(num_steps, device=log_probs.device)[None, :]
    log_probs_taken = log_probs[envs, steps, tau.actions]
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs_taken, entropy
    # END SOLUTION


tests.test_compute_logprobs_and_entropy(compute_logprobs_and_entropy, PolicyNetwork)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Building up to the loss function

We need to compute the probability ratio $\pi(a_t | s_t) / \pi_{old}(a_t | s_t)$ for each timestep taken in the rollout. This is used to compute the importance weights $\text{iw}_t$, which allows us to learn off-policy. If `args.clip_coef` is not none, we also clamp the importance weights between `1 - args.clip_coef` and `1 + args.clip_coef`.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `compute_importance_weights`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

Keep the result numerically stable by exponentiating the difference between the logprobs.
Gradients should **NOT** flow through the importance weights. Make sure to use `.detach()` to prevent this.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compute_importance_weights(logprobs_taken, tau: RolloutTensors, clip_coef: Optional[float]) -> t.Tensor:
    """
    Compute importance weights from log probabilities.

    Keeps the result numerically stable by exponentiating the difference between logprobs.
    Gradients should NOT flow through the importance weights (uses .detach()).
    Optionally clips the weights to [1 - clip_coef, 1 + clip_coef].
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    iw = t.exp(logprobs_taken - tau.logprobs).detach()  # Detach to prevent gradient flow
    if clip_coef is not None:
        iw = t.clamp(iw, 1 - clip_coef, 1 + clip_coef)
    return iw
    # END SOLUTION


tests.test_compute_importance_weights(compute_importance_weights)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `normalize_returns`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 5 minutes on this exercise.
> ```

Normalize the returns by ensuring zero mean, unit variance **across all trajectories and timesteps**. Don't overthink this one, should be a one-liner.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def normalize_returns(returns: Float[Tensor, " num_envs num_steps"]) -> Float[Tensor, " num_envs num_steps"]:
    """
    Normalizes the returns by ensuring zero mean, unit variance across all trajectories and timesteps.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    return (returns - returns.mean()) / (returns.std() + 1e-8)
    # END SOLUTION


tests.test_normalize_returns(normalize_returns)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `compute_reinforce_loss`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 5 minutes on this exercise.
> ```

This should be easy with everything else you've got.
The loss on timestep $t$ is 
$$
\text{iw}_t \log \pi(a_t | s_t) \big( G_t - b(s_t) \big)
$$
where $G_t$ is the return (already normalized by the caller), $\text{iw}_t$ is the importance weight, and $\log \pi(a_t | s_t)$ are the logprobs, each for timestep $t$. The normalization (mean-zero, unit-variance across all timesteps and trajectories) that acts as the baseline is applied in `compute_loss` before calling this function, controlled by `args.normalize_returns`. PPO uses a learned baseline called a critic, which we will see tomorrow. For now, the critic $b(s_t)$ is simply the average return for each trajectory, which we have already done in `compute_returns`.

The total loss is the mean of the losses over all timesteps, over all trajectories.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compute_reinforce_loss(
    returns: Float[Tensor, " num_envs num_steps"],
    logprobs_taken: Float[Tensor, " num_envs num_steps"],
    iw: Float[Tensor, " num_envs num_steps"],
) -> Float[Tensor, ""]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    adv = returns - returns.mean(dim=0, keepdim=True)   # baseline per timestep across envs
    return (iw * logprobs_taken * adv.detach()).mean()
    # END SOLUTION


tests.test_compute_reinforce_loss(compute_reinforce_loss)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Live training visualisation (optional)

To *watch* training, `utils.py` provides two helpers that render the first 16 environments of a
rollout as a 4x4 grid of cartpoles: `utils.rollout_grid_frames(obs)` returns the raw `(T, H, W, 3)`
frames, and `utils.render_rollout_grid_html(obs)` encodes them as a single autoplaying/looping MP4
(one ffmpeg encode, ~0.2s). The trainer logs these every `video_log_freq` rollouts — to wandb if
`use_wandb`, and/or inline if `live_viz` via `VPGTrainer._log_video` below.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Trainer

This is the function that will handle the full training loop. We've provided you with the template of a training loop which should be very similar to yesterday's.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `VPGTrainer`

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵🔵
> 
> You should spend up to 45 minutes on this exercise.
> ```

You should fill in the following methods. Ignore logging, can just copy from the solution later.

* `compute_loss` - this method should compute the loss for the VPG objective function.

The training loop is rather standard once everything else is done: we do a rollout, we cut the result into batches, compute the loss, and update the weights from each batch, so we've provided it for you.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

from part2_q_learning_and_policy_gradient.probe import Probe4, Probe5

class VPGTrainer:
    def __init__(self, args: VPGArgs):
        set_global_seeds(args.seed)
        self.args = args

        device = args.device

        self.rng = t.Generator(device=device).manual_seed(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        if args.env_id == "CartPole-gpu":
            self.envs = CartPole(num_envs=args.num_envs, device=device)
        elif args.env_id == "Probe4-v0":
            self.envs = Probe4(args.num_envs)
        elif args.env_id == "Probe5-v0":
            self.envs = Probe5(args.num_envs)
        else:
            raise ValueError(f"Environment {args.env_id} not supported")

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        self.num_envs = args.num_envs
        self.action_shape = self.envs.action_space.shape
        self.num_actions = self.envs.action_space.n
        self.obs_shape = self.envs.observation_space.shape

        # Create our networks & optimizer
        self.policy_network = PolicyNetwork(self.obs_shape, self.num_actions).to(device)

        # Compile the policy network for faster inference
        if self.args.compile:
            self.policy_network = t.compile(self.policy_network)

        self.optimizer = t.optim.Adam(self.policy_network.parameters(), lr=args.lr, eps=1e-5, maximize=True)
        self.optimizer.zero_grad()

        # Create our agent
        self.agent = VPGAgent(envs=self.envs, policy_network=self.policy_network, args=self.args, rng=self.rng)

    def compute_loss(self, tau: RolloutTensors) -> tuple[t.Tensor, dict[str, Any]]:
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        returns = compute_returns(tau.rewards, tau.dones, self.args.gamma)  # (num_envs, timestep)

        if self.args.normalize_returns:
            returns = normalize_returns(returns)

        logprobs_taken, entropy = compute_logprobs_and_entropy(tau, self.policy_network)

        iw = compute_importance_weights(logprobs_taken, tau, self.args.clip_coef) if self.args.use_iw else t.ones_like(logprobs_taken)
        r_joy = compute_reinforce_loss(returns, logprobs_taken, iw)
        avg_entropy = entropy.mean()

        joy = r_joy + self.args.ent_coef * avg_entropy

        # END SOLUTION

        info = {
            "entropy": avg_entropy.item(),
            "r_joy": r_joy.item(),
            "iw": iw.mean().item() if self.args.use_iw else None,
        }

        return joy, info

    def _log_video(self, rollout: "Rollout", avg_lifespan: float, step: int):
        """Render the rollout's first 16 envs as a 4x4 cartpole grid and log it. This is how
        `video_log_freq` works for VPG: the env is the batched GPU CartPole (no gym RecordVideo),
        so we render the rollout we already have. Logs to wandb if use_wandb, and/or displays it
        inline if live_viz (notebook). Reuses the rollout, so no extra env steps."""
        if not (self.args.use_wandb or self.args.live_viz):
            return  # nowhere to send it; skip the work
        try:
            tau = rollout.get()
            obs, dones = tau.obs, tau.dones  # (num_envs, T, 4), (num_envs, T)
            if self.args.use_wandb:
                frames = utils.rollout_grid_frames(obs, dones=dones)  # (T, H, W, 3)
                # wandb.Video wants (T, C, H, W)
                wandb.log({"rollout_video": wandb.Video(frames.transpose(0, 3, 1, 2), fps=50)}, step=step)
            if self.args.live_viz:
                from IPython.display import clear_output, display
                clear_output(wait=True)
                print(f"rollout {rollout.timestep} steps | avg lifespan "
                      f"{avg_lifespan:.1f}/{self.args.num_steps_per_rollout}")
                display(utils.render_rollout_grid_html(obs, dones=dones))
        except Exception as e:  # never let visualization break training
            print(f"[video log skipped: {e}]")

    def update_learning_rate(self, time_steps, args):
        if args.use_lr_decay and args.lr_frac > 0:
            progress = min(1.0, max(time_steps / args.total_timesteps, 0) / args.lr_frac)
            return (progress * args.lr_end) + ((1 - progress) * args.lr)
        return args.lr

    def train(self) -> None:
        """
        Trains the agent by generating rollouts and updating the policy.
        The progress bar tracks total environment steps.
        """
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
            )
            wandb.watch(self.policy_network, log="all", log_freq=50)

        # --- Setup ---
        rollout = Rollout(
            num_envs=self.num_envs,
            max_steps=self.args.num_steps_per_rollout,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            device=self.args.device,
        )

        # Calculate the total number of rollouts to perform
        env_steps_per_rollout = self.args.num_steps_per_rollout * self.args.num_envs
        num_updates = self.args.total_timesteps // env_steps_per_rollout
        train_steps = 0  # Counter for gradient updates

        # --- Training Loop ---
        # The progress bar is managed manually with a `with` statement.
        # `total` is set to the total environment steps we want to run.
        # The loop iterates `num_updates` times, not `total_timesteps` times.
        with tqdm(
            total=self.args.total_timesteps,
            unit=" env steps",
            unit_scale=True,
            desc="Training",
            miniters=1,
            mininterval=0.02,
        ) as pbar:
            env_steps_consumed = 0

            for update_num in range(num_updates):
                # 1. Generate a new rollout from the environment

                rollout, agent_info = self.agent.gen_rollout(rollout)

                # 2. Split the rollout into batches along the num_envs dimension

                rollout_batches = rollout.get_batches(self.args.batch_size)

                # 3. Logging and Progress Bar Update
                # This part is outside the inner loop to only log once per rollout
                avg_lifespan = agent_info["lifespan"].float().mean().item()
                std_lifespan = agent_info["lifespan"].float().std().item()
                max_lifespan = agent_info["lifespan"].max().item()

                # Log a 4x4 grid video of the rollout every `video_log_freq` rollouts. This is what
                # video_log_freq means for VPG (the GPU env has no gym RecordVideo) — see _log_video.
                if self.args.video_log_freq and (update_num % self.args.video_log_freq == 0):
                    self._log_video(rollout, avg_lifespan, step=env_steps_consumed)

                if (avg_lifespan + 0.5) > self.args.num_steps_per_rollout and std_lifespan < 0.01:
                    print("Agent has learned to play optimally!")
                    if self.args.video_log_freq:
                        self._log_video(rollout, avg_lifespan, step=env_steps_consumed)
                    break

                # 4. Advance env-step counter before gradient updates (one rollout collected)
                env_steps_consumed += self.args.num_steps_per_rollout * self.args.num_envs

                # 5. For each batch, perform multiple gradient updates
                for i in range(self.args.rollout_use_count):
                    for batch in rollout_batches:
                        loss, reinforce_info = self.compute_loss(batch)

                        info = {**agent_info, **reinforce_info}

                        loss.backward()
                        # clip_grad_norm_ returns the total (pre-clip) grad norm, so a single call
                        # both clips and gives us the value to log — no redundant second pass.
                        max_norm = self.args.max_grad_norm if self.args.max_grad_norm is not None else float("inf")
                        grad_norm = t.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=max_norm)

                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        train_steps += 1

                        new_lr = self.update_learning_rate(env_steps_consumed, self.args)

                        for pg in self.optimizer.param_groups:
                            pg["lr"] = new_lr

                        # Create info string to display in the progress bar
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        info_dict = {
                            "joy": f"{info['r_joy']:.4f}",
                            "traj_len": f"{avg_lifespan:.2f} ± {std_lifespan:.2f} (max: {max_lifespan:.2f})",
                            "H": f"{info['entropy']:.4f}",
                            "iw": f"{info['iw']:.4f}" if self.args.use_iw else None,
                            "∇": f"{grad_norm:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }

                        pbar.set_postfix(info_dict)

                # Progress bar advances once per rollout (env steps actually collected)
                pbar.update(self.args.num_steps_per_rollout * self.args.num_envs)

        # --- Cleanup ---
        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


tests.test_compute_loss(VPGTrainer, VPGArgs, Rollout)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Probes

As yesterday, we will be using probes to test our model. They've been implemented for you.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network
    args = VPGArgs(
        env_id=f"Probe{probe_idx}-v0",
        wandb_project_name=f"test-probe-{probe_idx}",
        total_timesteps=[7500, 7500, 12500, 10000, 10000][probe_idx - 1],
        lr=5e-3,
        num_envs=4,
        video_log_freq=None,
        use_wandb=False,
        device="cpu",
        ent_coef=0.0,
        clip_coef=None,
        normalize_returns=False,
        rollout_use_count=1,
        show_probs=True,
    )
    trainer = VPGTrainer(args)
    trainer.train()
    agent = trainer.agent

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[args.gamma], [1.0]],
        [[1.0]],
        [[1.0], [1.0]],
    ]
    expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    tolerances = [1e-3, 1e-3, 1e-3, 2e-3, 2e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(args.device)

    # Calculate the actual value & probs, and verify them
    with t.inference_mode():
        probs = agent.policy_network(obs).softmax(-1)
    expected_probs = expected_probs_for_probes[probe_idx - 1]
    if expected_probs is not None:
        print(f"Probs: {probs}")
        print(f"Expected probs: {t.tensor(expected_probs).to(args.device)}")
        t.testing.assert_close(probs, t.tensor(expected_probs).to(args.device), atol=tolerances[probe_idx - 1], rtol=0)
    print(f"Probe {probe_idx} tests passed!\n")


gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)
gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)

if MAIN:
    for probe_idx in [4, 5]:
        test_probe(probe_idx)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Training Run

Vanilla Policy Gradient can often be a bit finicky and unstable to train (which is why in practice we use PPO instead).
None-the-less, I've tried to find a good set of hyperparameters such that it trains in a minute or so.
Running this should cause a grid of 4x4 videos to render in the notebook. The background flashes pink when
the agent dies and is reset so you can easily see when a new episode starts.

Set
```python
live_vis = True
```
if you want to see videos of the agetn as it trains in-line.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    device = t.device("cuda")

    args_fast = VPGArgs(
        use_wandb=False,
        num_envs=512,
        num_batches_per_rollout=1,
        total_timesteps=50_000_000,
        num_steps_per_rollout=500,
        rollout_use_count=1,  # this seems to matter a lot
        ent_coef=0.0,  # didn't need this all along
        clip_coef=0.1,  # can sometimes work with no clipping, but it helps
        max_grad_norm=1,
        normalize_returns=False,
        lr=1e-3,  # risky!
        use_lr_decay=True,
        use_iw=False,  # dont' need it if we only use each rollout once in one
        lr_end=1e-3,
        lr_frac=0.6,
        compile=False,
        gamma=0.99,
        seed=1337,
        device=device,
        video_log_freq=10,
        live_viz=False,
    )
    trainer = VPGTrainer(args_fast)
    trainer.train()

