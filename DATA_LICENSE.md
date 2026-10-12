# Data license and attribution

The source code in this repository is covered by the root `LICENSE` file. The
dataset is a separate work and is **not relicensed under MIT**.

The frozen experimental splits used by this project were derived from the UBC-NLP
`nilechat-arabizi-mor` resource, which is distributed under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0). Users must
also review and accept the access conditions shown on the upstream Hugging Face
dataset page before using the source resource:

- Dataset: https://huggingface.co/datasets/UBC-NLP/nilechat-arabizi-mor
- License: https://creativecommons.org/licenses/by-nc/4.0/
- Requested citation: El Mekki et al., "NileChat: Towards Linguistically Diverse
  and Culturally Aware LLMs for Local Communities," EMNLP 2025.

The Arabizi side was generated from Arabic-script Moroccan material with
`CohereLabs/c4ai-command-r-plus-08-2024`; it is not a collection of naturally
typed Arabizi. The full CSV exports are deliberately not distributed in this
repository. `data/split_ids.csv` contains only the identifiers assigned to each
experimental split, while `data/README.md` records the expected file hashes and
local placement. Users must obtain the source data through the upstream access
process. Reuse must remain noncommercial, preserve attribution, and comply with
the upstream access terms.

## Manuscript

The MIT license applies to the project code and its software documentation.
The camera-ready manuscript, LaTeX source, bibliography, and paper figures are
provided for scholarly reference and are excluded from that software license.
Their availability does not establish a separate open-content license or change
any publisher agreement. Cite the paper using `CITATION.cff`.
