# Camera-ready source

`icalp2026_paper.tex` and `custom.bib` are the sources for the camera-ready PDF
stored at the repository root. The manuscript uses Springer's LNCS/CCIS class
(`llncs`) and standard LaTeX packages declared in the source.

With an LNCS-capable TeX installation, compile from this directory using:

```text
pdflatex icalp2026_paper.tex
bibtex icalp2026_paper
pdflatex icalp2026_paper.tex
pdflatex icalp2026_paper.tex
```

Generated `.aux`, `.bbl`, `.blg`, `.log`, and PDF files in this directory are
ignored. The root PDF remains the public accepted manuscript. After publication,
update the repository citation with the official Springer DOI and page numbers.
