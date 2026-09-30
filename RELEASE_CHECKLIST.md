# Public release checklist

## Completed

- [x] Camera-ready PDF and matching LaTeX source included.
- [x] README updated to the four-system camera-ready results.
- [x] Dataset provenance, generated-Arabizi scope, and CC BY-NC 4.0 notice added.
- [x] Notebook outputs and local temporary paths removed.
- [x] mT5 protocol, portable runner, run metadata, and utility tests included.
- [x] Evaluation commands accept explicit paths instead of local absolute paths.
- [x] Raw checkpoints, caches, reviewer workbooks, and internal QA renderings excluded.
- [x] Aggregate human-annotation outcomes and final adjudication counts recorded.
- [x] Machine-readable project citation added.

## Confirm before pushing `main`

- [x] Remove the gated CSV exports from the working tree and all Git history;
      retain verified local copies and publish only split IDs and hashes.
- [ ] Review the complete staged diff and ensure no private annotation file or
      model checkpoint is present.
- [ ] Fetch `origin/main` and confirm that it has not advanced.

## Update after publication

- [ ] Replace the accepted-manuscript wording with the official proceedings URL.
- [ ] Add DOI, page numbers, publisher metadata, and the final Springer citation.
