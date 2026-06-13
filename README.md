# iliad2-aixi

Solomonoff Induction & AIXI worksheets and slides for the **Iliad Intensive**
(London Initiative for Safe AI), April 2026.

## Built PDFs

Every push to `main` rebuilds all PDFs and publishes them to a one-page site:

**https://davidquarel.github.io/iliad2-aixi/**

The site links to each worksheet with and without solutions, and the slides in
both presentation (with `\pause` reveals) and handout (collapsed) form. No PDFs
are committed — they live in GitHub's Pages storage and are replaced on each
push. CI lives in `.github/workflows/build.yml`.

## Local build

`./build.sh` builds the worksheets with and without solutions. For slides,
`pdflatex iliad-david1-slides.tex` gives the presentation (pauses on); add the
handout flag to collapse reveals:
`pdflatex "\def\HANDOUT{}\input{iliad-david1-slides}"`.
