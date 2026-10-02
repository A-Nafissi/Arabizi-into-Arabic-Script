# Camera-ready source

`icalp2026_paper.tex` and `custom.bib` are the sources for the camera-ready PDF
stored at the repository root. The manuscript uses Springer's LNCS/CCIS class
(`llncs`) and standard LaTeX packages declared in the source. Keep the supplied
`figures/ngram_precision.png` alongside the source when compiling.

With an LNCS-capable TeX installation, compile from this directory using:

```text
pdflatex icalp2026_paper.tex
bibtex icalp2026_paper
pdflatex icalp2026_paper.tex
pdflatex icalp2026_paper.tex
```

Generated `.aux`, `.bbl`, `.blg`, `.log`, and PDF files in this directory are
ignored. The root PDF remains the author's accepted manuscript; repository
access is available on request while the repository is private. After publication,
update the repository citation with the official Springer DOI and page numbers.

The 2026-10-02 revision documents the recovered preprocessing and seed-42 split reconstruction. The reference-audit kappa remains undefined; 100% observed agreement is reported. The root PDF was rebuilt from this source and visually checked.
