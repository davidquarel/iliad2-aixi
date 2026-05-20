# Candidate exercises for the AIXI worksheet

Brainstormed ideas not currently in `aixi-worksheet.tex`. Each entry: statement,
why it earns its place, proof sketch, imports, difficulty.

The worksheet already covers: chain rule for $\nu^\pi$, posterior update,
one-step predictive of $\xi$, bounded value, multi-step posterior linearity,
sup=max, finite/infinite-horizon Bellman, backward induction, $V_\nu^*$
existence, dominance, finite/infinite linearity of $V_\xi$, convexity and
non-linearity of $V_\xi^*$, TV bound on value difference, covering, on-policy
value convergence, AIXI cannot be fooled in deterministic environments,
likelihood-ratio martingale, change of measure, self-optimizing on finite
model classes.

---

## A1. Pareto optimality of AIXI  [15]  ⭐

**Statement.** There is no policy $\pi'$ such that
$V_\nu^{\pi'} \ge V_\nu^{\pi^*_\xi}$ for all $\nu\in\mathcal M$
with strict inequality for some $\nu$.

**Why.** A real theorem, not a lemma — and short. Says the Bayes-optimal
policy is undominated across the entire class. Natural follow-on to the
finite-horizon linearity exercise (`prob:linearity_finite`).

**Sketch.** Suppose such $\pi'$ exists. By linearity of $V_\xi^{\cdot}$ in $\nu$,
$V_\xi^{\pi'} = \sum_\nu w_\nu V_\nu^{\pi'} \ge \sum_\nu w_\nu V_\nu^{\pi^*_\xi}
= V_\xi^{\pi^*_\xi}$, with at least one strict inequality. So
$V_\xi^{\pi'} > V_\xi^{\pi^*_\xi}$ — contradicting optimality of $\pi^*_\xi$ for
$\xi$.

**Imports.** Linearity of $V_\xi^\pi$ in $\nu$ — already proved in
`prob:linearity_finite` / `prob:linearity_inf`.

---

## A2. Heaven-and-hell: AIXI is not asymptotically optimal in general  [20]  ⭐

**Statement.** Exhibit a two-environment class where, with positive
probability under the true $\mu$, AIXI commits to the wrong action forever
and never explores enough to identify $\mu$.

Concrete construction: two deterministic environments
$\mathcal M = \{\mu_H, \mu_L\}$ both with two actions $\{a_1, a_2\}$.
- Action $a_1$: gives reward $1/2$ in both environments.
- Action $a_2$: gives reward $1$ in $\mu_H$ ("heaven"), $0$ in $\mu_L$ ("hell").
- Once $a_2$ is played, the environments are revealed by the reward.

Show: for any prior $w$, AIXI plays $a_1$ forever, never distinguishing the
two environments. So if $\mu = \mu_H$, AIXI achieves reward $1/2$ per step
while the optimal policy under $\mu_H$ achieves $1$.

**Why.** A *negative* result — counterweight to the positive convergence
theorems already in the sheet. Workshop participants typically over-update
toward "AIXI is magic"; this lands the cautionary point that AIXI is only
optimal in a Bayesian-average sense, not pointwise. Also a great hands-on
exercise in computing the Bayes-optimal policy by direct calculation.

**Sketch.** Compute $V_\xi^{a_1}$ and $V_\xi^{a_2}$ from the prior $w$ and
the discount $\gamma$:
- $V_\xi^{a_1}$ playing $a_1$ forever: $1/2 \cdot 1/(1-\gamma)$.
- $V_\xi^{a_2}$ at the root: $w(\mu_H)\cdot 1/(1-\gamma) + w(\mu_L)\cdot 0$.

If $w(\mu_H) < 1/2$, the agent prefers $a_1$. Crucially, playing $a_1$ yields
percepts consistent with *both* environments, so the posterior is unchanged
forever. The agent never plays $a_2$, never learns. No imported results
needed beyond the value-function definition.

**Imports.** None — only `def:value` and basic geometric-series.

---

## A3. Posterior concentration in AIXI  [15]

**Statement.** Along the $\mu^\pi$-trajectory, the posterior weight on the
true environment $w(\mu \mid \aes_{<t})$ converges a.s. to some
$W_\infty \in [w_\mu, 1]$. Under an identifiability hypothesis
(per-step expected KL between $\mu$ and every other $\nu \in \mathcal M$ has
positive long-run rate), $W_\infty = 1$ a.s.

**Why.** The AIXI analogue of the Solomonoff posterior-concentration
exercise (Solomonoff sheet S1). The current AIXI sheet's self-optimizing
problems work with the likelihood-ratio martingale $X_{\nu,t}$ —
this exercise is a clean corollary nobody states explicitly. Closes the
"AIXI learns" narrative.

**Sketch.** Recall $X_{\nu,t} := \nu^\pi(\aes_{<t})/\mu^\pi(\aes_{<t})$
is a $\mu^\pi$-martingale (already shown in `prob:martingale`) and bounded
below by $0$. By Doob, $X_{\nu,t} \to X_{\nu,\infty} < \infty$ a.s.
Posterior weight $w(\mu \mid \aes_{<t}) = w_\mu \cdot 1 \,/\, \sum_\nu w_\nu X_{\nu,t}$
(since $X_{\mu,t} \equiv 1$). Bounded denominator gives convergence.
Identifiability forces $X_{\nu,\infty} = 0$ for $\nu \ne \mu$ by SLLN on the
log-likelihood-ratio.

**Imports.** Doob's martingale convergence (already used in
`prob:X_converges`), SLLN.

---

## A4. Value of perfect information is nonnegative  [10]

**Statement.** Let $\pi^*_\xi$ be Bayes-optimal under the prior, and let
$\pi^*_\nu$ be optimal under each $\nu$. Then
$$\sum_\nu w_\nu V_\nu^{\pi^*_\nu} \;\ge\; V_\xi^{\pi^*_\xi}.$$

**Why.** Short, conceptually crisp. The gap on the LHS minus RHS is exactly
the *value of perfect information* — what an agent would pay to be told the
true environment. Reframes A1 (Pareto) from a different angle. Quick to
state, quick to prove, very satisfying punchline.

**Sketch.** By linearity, $V_\xi^{\pi^*_\xi} = \sum_\nu w_\nu V_\nu^{\pi^*_\xi}$.
For each $\nu$, optimality of $\pi^*_\nu$ for $\nu$ gives
$V_\nu^{\pi^*_\nu} \ge V_\nu^{\pi^*_\xi}$. Sum with weights $w_\nu$.

**Imports.** Linearity (`prob:linearity_finite` / `prob:linearity_inf`).

---

## A5. Effective horizon for $\gamma$-discounting  [12]

**Statement.** With geometric discount $\gamma\in[0,1)$ and bounded rewards
$r \in [0, r_{\max}]$, any two policies differ in value by at most
$r_{\max}\,\gamma^h/(1-\gamma)$ when their interaction differs only after time $h$.
Conclude that $h$-step lookahead is $\varepsilon$-optimal for
$h = \log_{1/\gamma}\bigl(r_{\max}/(\varepsilon(1-\gamma))\bigr)$.

**Why.** Connects the abstract infinite-horizon value function to actual
planning. Makes the infinite-horizon limit feel less scary, and shows where
the discount factor's "memory length" comes from. Standard MDP fact, worth
doing once in the AIXI notation.

**Sketch.** Geometric tail $\sum_{t \ge h} \gamma^t r_{\max} = r_{\max}\gamma^h/(1-\gamma)$.
Solve for $h$.

**Imports.** None.

---

## A6. Wireheading: reward maximization vs. utility maximization  [15]

**Statement.** If the reward signal $r_t$ is part of the percept $e_t$ and
the agent can take an action that directly modifies the reward channel
(e.g. a "wirehead" action that sets $r_t = r_{\max}$ for all $t$ thereafter),
then the Bayes-optimal AIXI policy will take that action whenever it
is available, regardless of what "real" utility the original reward signal
was supposed to track.

**Why.** Bridges to the AGI-safety chapter (`agi_safety.tex`,
`agency.tex`). Conceptually loaded but mathematically a 10-line proof.
Lands the point that AIXI optimizes *its perceived reward stream*, not any
external notion of utility.

**Sketch.** $V_\xi^\pi$ depends *only* on the rewards in the percept stream.
Therefore any policy whose reward-stream distribution stochastically
dominates $\pi$'s — including a policy that hijacks the register to output
$r_{\max}$ forever — has at-least-equal value. The wirehead policy
achieves the maximum possible value $r_{\max}/(1-\gamma)$.

**Imports.** Definitions only.

---

## Top picks

If only adding 2: **A1** (Pareto, [15]) and **A2** (heaven-and-hell, [20]).
A1 is a quotable positive result that costs almost nothing given the
linearity exercise already in the sheet. A2 is the standard cautionary
example that prevents over-claiming about AIXI's guarantees.

A3 (posterior concentration) is the natural Bayesian-consistency capstone
to the self-optimizing chapter, almost free given the martingale machinery
already built. A4 (value of information) is a [10] aside that pairs
beautifully with A1. A6 is a great workshop discussion piece — short proof,
big philosophical payoff — if the audience is safety-curious.
