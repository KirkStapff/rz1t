# Technical report

Source: [`main.tex`](main.tex). Compiled PDF: [`main.pdf`](main.pdf).

Figures are generated into `../reference/paper-figures/` by:

```bash
uv run python scripts/make_paper_figures.py
```

To compile a PDF (TeX Live / MacTeX / tectonic):

```bash
cd paper
tectonic -X compile main.tex
# or:
pdflatex main.tex
pdflatex main.tex
```
