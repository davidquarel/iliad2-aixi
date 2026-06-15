# iliad2-aixi

Worksheets, slides, and interactive Colab notebooks for the **Iliad Intensive**
(London Initiative for Safe AI), April 2026 — covering Policy Gradients &
Misgeneralization (D.2) and Solomonoff Induction & AIXI (D.3).

## Built PDFs

Every push to `main` rebuilds all PDFs and publishes them to a one-page site:

**https://davidquarel.github.io/iliad2-aixi/**

The site links to each worksheet with and without solutions, and the slides in
both presentation (with `\pause` reveals) and handout (collapsed) form. No PDFs
are committed — they live in GitHub's Pages storage and are replaced on each
push. CI lives in `.github/workflows/build.yml`.

## Local build

**Slides** — any deck (`iliad-david1-slides`, `vpg-slides`, `goalmisgen-slides`):

```bash
pdflatex goalmisgen-slides.tex                       # presentation (\pause reveals on)
pdflatex "\def\HANDOUT{}\input{goalmisgen-slides}"   # handout (reveals collapsed)
```

Decks that cite (`iliad-david1-slides`, `goalmisgen-slides`) need a `bibtex <deck>` pass
between two `pdflatex` runs; `vpg-slides` has no citations.

**Worksheets** — `solomonoff-all`, `aixi-worksheet`: same `pdflatex` (run twice for cleveref).
Toggle solutions with the `\solutionstrue` / `\solutionsfalse` line near the top of the `.tex`.

**Notebooks** — regenerate the Colab notebooks from their `gen/masters/master_*.py` sources:

```bash
pip install -r gen/requirements-gen.txt        # once
python gen/core/main.py --chapters='2.*'       # 2.2 + 2.6 -> build/exercises/<part>/*.ipynb (gitignored)
```

CI publishes the generated notebooks to the `notebooks` branch, where Colab opens them. To run
one end-to-end locally, execute it with `jupyter nbconvert --execute` and the support dir on
`PYTHONPATH` (see `.github/workflows/build.yml`).
