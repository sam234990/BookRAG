# Third-Party Notices

This document identifies important third-party software and source material
used by or referenced from BookRAG. It is not necessarily an exhaustive list
of all transitive Python dependencies.

## MinerU

BookRAG uses MinerU 2.1.11 for PDF parsing and document information
extraction.

- Project: <https://github.com/opendatalab/MinerU>
- Version used by BookRAG: 2.1.11
- Release commit: `30b698ecc51fd04e33ba4650e918dd9f8fe5adbe`
- License at that release: GNU Affero General Public License v3.0
- License text at the release commit:
  <https://github.com/opendatalab/MinerU/blob/30b698ecc51fd04e33ba4650e918dd9f8fe5adbe/LICENSE.md>

Some BookRAG integration code may be adapted from MinerU example code.
Relevant files are marked `SPDX-License-Identifier: AGPL-3.0-only` and retain
a source reference to the applicable upstream implementation whenever it can
be identified reliably.

## Ultralytics

The BookRAG dependency set includes Ultralytics. Ultralytics is available
under AGPL-3.0 unless the user has obtained a separate enterprise license.

- Project: <https://github.com/ultralytics/ultralytics>
- License information: <https://github.com/ultralytics/ultralytics#license>

Users are responsible for complying with the applicable Ultralytics license.

## DocLayout-YOLO

The BookRAG dependency set includes `doclayout_yolo`, directly or through the
MinerU parsing pipeline.

- Project: <https://github.com/opendatalab/DocLayout-YOLO>

Users should review the version-specific license and model terms before
commercial deployment or redistribution.

## DROP evaluation-derived code

`Eval/utils/m3doc_eval.py` contains code copied and subsequently modified
from the DROP evaluation implementation referenced at:

<https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py>

The upstream repository does not currently expose a detected repository
license. This file is therefore excluded from the BookRAG Apache-2.0 and
AGPL-3.0 software grants. The exclusion does not resolve the upstream
provenance question. Before relicensing or redistributing this file under a
BookRAG license, the maintainer should locate the original applicable license,
obtain permission, or replace the copied implementation with an independently
written implementation based on the public metric specification.

## Knowledge graph extraction implementation

`Core/pipelines/kg_extractor.py` was independently authored for BookRAG.
GraphRAG, nano-graphrag, and LightRAG were consulted as conceptual references;
no source code from those projects was copied into that file. The existing
reference links are retained in its documentation.

## LightRAG-derived entity-extraction prompt

Part of `Core/prompts/kg_prompt.py`, specifically the entity type defaults,
delimiters, and entity-extraction prompt block through
`ENTITY_IF_LOOP_EXTRACTION`, is adapted from the LightRAG entity-extraction
prompt. BookRAG modifies the instructions and examples; the rest of the file
was independently authored for BookRAG.

- Project: <https://github.com/HKUDS/LightRAG>
- Copyright: Copyright (c) 2025 LightRAG Team
- License: MIT License
- Fixed upstream prompt revision:
  <https://github.com/HKUDS/LightRAG/blob/da46b341dc1b2c6c578439374ed45a30bea493db/lightrag/prompt.py>
- MIT notice: [`LICENSES/LightRAG-MIT.txt`](./LICENSES/LightRAG-MIT.txt)

The prompt was also used in the BookRAG authors' prior GraphRAG work and was
subsequently modified for BookRAG. That intermediate source is retained here
for provenance:

<https://github.com/JayLZhou/GraphRAG/blob/4e87938e46f90f3616fb27f955e8b2dc43743bde/Core/Prompt/EntityPrompt.py>

The combined file is marked:

`SPDX-License-Identifier: MIT AND (Apache-2.0 OR AGPL-3.0-only)`
