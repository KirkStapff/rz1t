# Technical report

Source: [`main.tex`](main.tex).

Figures are generated into `../reference/paper-figures/` by:

```bash
uv run python scripts/make_paper_figures.py
```

To compile a PDF (TeX Live / MacTeX):

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```
