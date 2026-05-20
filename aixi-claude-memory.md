# AIXI Exercise Sheet — Session Summary (2026-04-09)

## What was built

A self-contained exercise sheet (`iliad/iliad.tex`) for a one-day AIXI tutorial at the Iliad Intensive (LISA, April 2026). The goal: guide students with no AIXI background through a proof that **AIXI learns to act optimally** (the self-optimizing theorem).

Companion slides were also edited: `iliad/aixi/aixi.tex` — added a discounted value function frame and dropped commas in conditioning notation.

## Document structure (final)

**10 pages** without solutions, **20 pages** with. Solutions toggle via `\solutionstrue`/`\solutionsfalse`.

| Section | Content |
|---|---|
| **Setup** (p1) | Definitions 1-4: Spaces, Policy, Environment, Interaction measure |
| **Problem 0** (p2) | Properties of measures: factorization, chain rule |
| **Mixture & Value** (p2-3) | Definitions 5-7: Expectation, Mixture/posterior, Value function |
| **Problem 1** (p3) | Properties of mixture: posterior update, one-step predictive, bounded value |
| **Problem 2** (p4-7) | Existence of optimal policies: sup=max, Bellman equation, backward induction, infinite horizon (greedy policy via Bellman) |
| **Problem 3** (p7) | Dominance of Bayesian mixture |
| **Problem 4** (p7-8) | Linearity of value function (with dominated convergence theorem) |
| **Problem 5** (p8-9) | Total variation bound on expectations |
| **Convergence defs** (p9-10) | Covering, events on infinite histories, almost surely |
| **Problem 6** (p10) | On-policy value convergence (via Blackwell-Dubins) |
| **Problem 7** (p11) | AIXI can't be fooled in deterministic environments |
| **Problems 8-10 defs** (p12) | Self-optimizing, supermartingale, Doob's theorem |
| **Problem 8** (p12-14) | Likelihood ratios are supermartingales |
| **Problem 9** (p14-15) | Change of measure |
| **Problem 10** (p16-18) | Self-optimizing proof (chain of inequalities, case split, conclusion) |
| **Interpretation** (p18) | Summary of what was proved |
| **Appendix A** (p19) | Worked example: Bayesian mixture (coin prediction) |
| **Appendix B** (p19) | Worked example: Value function |

## Key design decisions

### Notation
- **ae notation**: History written as $\ae_{<t} = a_1 e_1 \ldots a_{t-1} e_{t-1}$, matching the UAI book
- **No double-bar notation**: $\nu(\ae_{1:t})$ instead of $\nu(e_{1:t} \| a_{1:t})$, matching the slides
- **Primed summation variables**: $\ae'$ for dummy/summation histories, $\ae$ for actual history
- **No primes in definitions**: Only in sums/expectations where disambiguation is needed
- **Book-style conditioning**: $\nu(\cdot \mid \ae_{<t} a_t)$ (no comma), matching the book
- **Iverson brackets**: $\llbracket P \rrbracket$ for predicates, defined in notation list

### Definitions
- **Just-in-time**: Definitions appear right before their first use, not front-loaded
- **Split Definition 4**: Interaction measure (Setup), Expectation (before Value), Probability of events (before Problem 6)
- **Covering** moved to "Convergence" section (only needed for Blackwell-Dubins)
- **No measure theory**: Probability on infinite histories defined via finite-prefix events + countable operations. No sigma-algebras, no "measurable."

### Value function
- **Geometric discounting** $\gamma \in (0,1)$ with $(1-\gamma)$ normalization giving $V \in [0,1]$
- **Finite-horizon** $V_\nu^{\pi,m}$ defined first, **infinite-horizon** as its limit
- Boundedness proved as an exercise (Problem 1.3), not stated in definition

### Proof architecture
- **Two independent paths**: Problems 0-6 (on-policy convergence via Blackwell-Dubins) and Problems 8-10 (self-optimizing via supermartingales + change of measure). Problem 7 is standalone.
- **Self-optimizing restricted to finite $\cM$**: Avoids "convergence of mixture tails" lemma needed for countable case.
- **Bellman equation**: Proved as Problem 2.2, used in backward induction (2.3) and infinite-horizon existence (2.4).
- **Blackwell-Dubins** and **Doob's theorem**: Stated without proof (orange fact boxes).
- **Dominated convergence for sums**: Stated as a theorem, used to push $\lim$ through $\sum_\nu$ in linearity proof.

### Pedagogical choices
- **Warm-ups split into two groups**: Problem 0 (measure properties, needs only Defs 1-4) and Problem 1 (mixture properties, needs Defs 5-7)
- **Worked example in appendix**: Referenced from definitions, not blocking the problem flow
- **Difficulty ratings**: Knuth scale [00]-[50] on every subpart
- **Verbose solutions**: Every algebraic step shown, no leaps of faith
- **cleveref**: All cross-references are clickable hyperlinks
- **Section/subsection numbering**: Problems are sections (0-10), subparts are subsections (2.1, 2.2, etc.)

## Files modified
- `iliad/iliad.tex` — the exercise sheet (main deliverable)
- `iliad/aixi/aixi.tex` — slides (added discounted value frame, dropped conditioning commas)

## Known issues / future work
- The Bellman equation solution (Problem 2.2) could have the Step 4 algebra tightened
- Problem 5 (TV bound) is used exactly once — could be inlined into Problem 6 to save space
- The "Problems 8-10" unnumbered preamble section interrupts the section numbering
- Appendix references render as "Appendix A" not "Worked Example A" — minor
- Could add more exercises on the coin-flip example (compute optimal policy, show convergence numerically)
