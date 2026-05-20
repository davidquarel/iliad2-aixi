# Iliad Intensive worksheets — working notes

Context for resuming work on the Solomonoff / AIXI exercise sheets.

## What these files are

- `solomonoff-worksheet.tex` — Solomonoff Induction exercises (Leon Lang).
- `aixi-worksheet.tex` — AIXI exercises (David Quarel). **Reference template**, read-only.
- Both target the *Iliad Intensive / London Initiative for Safe AI*, April 2026, a one-day workshop.
- **End goal:** eventually **merge the two sheets**. All structural changes to the
  Solomonoff sheet are made to align it with the AIXI sheet so the merge is mechanical.

Build: `pdflatex -interaction=nonstopmode -halt-on-error solomonoff-worksheet.tex`
**twice** (cleveref needs two passes). Solution toggle is `\solutionstrue` near the top;
both toggle states must compile (true ≈ 12 pp, false ≈ 9 pp). The sheet is
**deliberately citation-free with no bibliography** — never add `\cite`; cite
inline in prose instead. PDF is the deliverable; the whole `iliad2-aixi/` dir is
untracked in git.

## Key user decisions (do NOT re-litigate)

1. **No semimeasures anywhere.** Environments are proper *measures*; prior weights
   sum to *exactly 1*; computability/lower-semicomputability is glossed over
   exactly as the AIXI sheet does.
2. **Mirror AIXI definitions** structurally (predictive-conditional environments,
   posterior-predictive mixture). Labels aligned: `def:environment`, `def:mixture`,
   `def:prob` (AIXI uses `def:environment`/`def:mixture`).
3. **Generic `\xi` everywhere; `\xi_U` introduced exactly once**, in the grey
   "Solomonoff prior" box ("the Bayesian mixture `\xi` under this prior is the
   universal mixture, written `\xi_U`"). Results are prior-agnostic, so the body
   uses generic `\xi`; the Solomonoff specialization is by *prior*
   (`w_ν=2^{-K(ν)}`) in Problem 5, which never needs the `\xi_U` symbol. The
   UTM-defined Solomonoff *distribution* `M(x)=Σ_{p:U(p)=x*}2^{-ℓ(p)}` is **not
   discussed at all** — fully removed. (We investigated the `M ≐× ξ_U`
   equivalence: it is Levin's coding theorem, genuinely hard, deferred in the
   textbook; finiteness does NOT help — it makes the identity false, not easier.
   Decision: ignore `M` entirely.)
4. **Drop all index machinery** (`K(i)`, `ν_i`, `min{K(i):ν_i=μ}`, dominance
   constant `c`, the `eq:second:representation` equation). Use `K(ν)` directly =
   "length of the shortest program that computes ν"; do not formalize `K` further.

## Done (solomonoff-worksheet.tex)

- Setup rewritten to mirror AIXI: `def:prob` (probability dist + `Δ`), `def:environment`
  (predictive conditionals, chain-rule joint), `def:mixture` (titled "Bayesian
  mixture `ξ_U`", posterior-predictive form).
- New `\section{The mixture is a predictor}` (`\label{prob:one_step_M}`, rated **[10]**):
  derives `ξ_U`'s posterior-predictive form and that `ξ_U(·∣x_<t)` is a proper
  distribution. Mirrors AIXI `prob:one_step`. Resolved Leon's old line-440 TODO (deleted).
- KL defs: both `P,Q` probability distributions (the `0 ln 0/q`, `p ln p/0`
  conventions kept).
- Pinsker (`thm:pinsker`) restated for probability distributions (`q0+q1=1`);
  binary lemma + its **[15]** proof unchanged.
- Problem 4 (`prob:bound`): `S_∞^μ ≤ -ln w_μ`, standard KL/Pinsker via `prob:one_step_M`.
- Problem 5 (`prob:bound_explicit`): now a one-line **[05]** — substitute
  `w_μ=2^{-K(μ)}` into Problem 4 ⇒ `S_∞^μ ≤ K(μ) ln 2` (no `κ`). Matches book
  `thm:Solomonoff_bound`.
- Appendix A Pinsker proof shrunk (~67→~30 lines): kept log-bound lemma + `q_x=0`
  case; deleted AM–GM fact and the `Φ`-monotonicity argument.
- Grey box retitled "The Solomonoff prior" — `M`-distribution paragraph removed;
  now also the sole place `\xi_U` is introduced.
- Goal section fixed: deleted the redundant + mathematically wrong `S_∞^ξ`
  display (it had `ξ(x_t∣x_<t)` as the outer weight instead of `μ(x_<t)`); fixed
  the dangling-colon sentence; deleted the stale commented-out semimeasure/`M`/`κ` blocks.
- `\xi_U`→`\xi` globally; reworded so the body is generic `\xi` and `\xi_U` is
  introduced once in the grey box (item 3 below — now done).

Plan file (historical): `/home/david/.claude/plans/new-aixi-day-solomonoff-worksheet-tex-s-vivid-rainbow.md`

## Pending / next

- **The merge itself** (solomonoff + aixi). Not started. The two sheets now share
  definition style and label names; remaining work is structural integration.
- Optional, discussed but not added: a starred **[05]** exercise that a
  multiplicative constant between predictors becomes a harmless additive,
  horizon-independent term in the cumulative-KL bound. Only relevant if `M` is
  ever reintroduced — currently it is not.

## Source-of-truth references (read-only)

- `../uaib2/bayes_prediction.tex` — `def:solomonoff_prior` (ξ_U), `def:solomonoff_distribution`
  (M), `thm:solmononff_equivalence` (M ≐× ξ_U, proof deferred to Zvonkin:70),
  `thm:Solomonoff_bound` (`S_∞ ≤ K(μ)ln2`).
- `../uaib1/UAIBook2.tex` — `thUniM` (Levin universality), Problem `exxiex` [C20]
  (exact-equality refinement, *not* the hard theorem).

## Conventions / gotchas

- `\qedhere` must sit *inside* the final display (`\[ … \qedhere \]`), never after `\]`.
- **Sum-scope ambiguity.** When an equation mixes a summed term with a standalone
  term (e.g.\ `Σ_ν w_ν X_ν + Y`), write the standalone term *first*: prefer
  `Y + Σ_ν w_ν X_ν`. With the standalone trailing, a reader cannot tell whether
  it's inside or outside the sum's scope. Applies to all `Σ … + (standalone)`
  patterns, including Pythagorean / Bregman decompositions and any bound where a
  free term is added to a summed quantity. Restate identities in this order
  everywhere they appear (statements, solutions, downstream uses).
- Standing memory rules: re-evaluate Knuth difficulty ratings whenever exercises
  are added/changed; never fabricate citations (sheet is citation-free anyway).
- After edits always rebuild twice and `grep -nE 'semimeasure|semiprob|(^|[^A-Za-z\\])M\(|\\kappa'`
  on non-comment lines to confirm no regressions; check both solution toggles.
