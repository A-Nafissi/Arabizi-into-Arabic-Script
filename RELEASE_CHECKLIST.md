# Public release checklist

## Completed

- [x] Camera-ready PDF and matching LaTeX source included.
- [x] README updated to the four-system camera-ready results.
- [x] Dataset provenance, generated-Arabizi scope, and CC BY-NC 4.0 notice added.
- [x] Maintained training notebook outputs removed; sanitized preprocessing/archive retain only count and audit stream outputs.
- [x] mT5 protocol, portable runner, run metadata, and utility tests included.
- [x] Evaluation commands accept explicit paths instead of local absolute paths.
- [x] Raw checkpoints, caches, reviewer workbooks, and internal QA renderings excluded.
- [x] Aggregate human-annotation outcomes and final adjudication counts recorded.
- [x] Machine-readable project citation added.

## Completed before pushing `main`

- [x] Remove the gated CSV exports from the working tree and all Git history;
      retain verified local copies and publish only split IDs and hashes.
- [x] Review the complete staged diff and ensure no private annotation file or
      model checkpoint is present.
- [x] Historical data-removal rewrite completed previously; this update uses a normal fast-forward push after fetching `origin/main`.

## Update after publication

- [ ] Replace the accepted-manuscript wording with the official proceedings URL.
- [ ] Add DOI, page numbers, publisher metadata, and the final Springer citation.

## Recovered preprocessing update (2026-10-02)

- [x] Focused preprocessing, baseline, and maintained mT5 wrapper notebooks organized.
- [x] Sanitized historical Colab archive retained.
- [x] Split reconstruction matches frozen CSV contents, row order, IDs and canonical byte hashes.
- [x] Preprocessing and training protocol documented; manuscript split limitation corrected.
- [x] Undefined reference-audit kappa reported correctly.
- [x] Credentials removed from every uploaded notebook.
- [ ] Account owner must revoke the exposed Hugging Face token; revocation cannot be confirmed from this workspace.

## Open-source release review (2026-10-12)

- [x] Reviewed all 84 historical file blobs reachable before this release; no Hugging Face/GitHub/AWS credential patterns or private-key headers found.
- [x] Historical CSV paths contain split IDs and aggregate metrics only; no full corpus exports or trained weights are included.
- [x] Reviewed retained notebook outputs: preprocessing counts and audit summaries only.
- [x] Corrected stray text in the MIT license and clarified public code/data separation.
- [x] Retained camera-ready status and documented reproduction limitations.

The earlier token-revocation reminder above remains unresolved; a clean repository
scan does not establish whether a previously exposed token was revoked.

Validation for this release: 15 pytest regression/release tests and 7 baseline
utility tests passed. The 2 PyTorch/Transformers-dependent tests were not run
in this lightweight review environment. Training and benchmark experiments
were not rerun for these documentation and licensing changes.
