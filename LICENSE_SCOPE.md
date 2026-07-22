# BookRAG License Scope

Copyright (C) 2025-2026 Shu Wang.

## Independently authored BookRAG code

Except where otherwise noted, independently authored source files under
`Core/`, together with the `main.py` program entry point, are dual-licensed
under either:

- the Apache License 2.0; or
- the GNU Affero General Public License v3.0 only.

Users may select either license when reusing those files independently.
The applicable SPDX expression is:

`Apache-2.0 OR AGPL-3.0-only`

## Mixed-origin prompt code

`Core/prompts/kg_prompt.py` contains an entity-extraction prompt block adapted
from MIT-licensed LightRAG material, together with independently authored
BookRAG prompt code. Its applicable SPDX expression is:

`MIT AND (Apache-2.0 OR AGPL-3.0-only)`

The LightRAG copyright and MIT permission notice are preserved in the source
file, [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md), and
[`LICENSES/LightRAG-MIT.txt`](./LICENSES/LightRAG-MIT.txt).

## Default complete BookRAG system

The default BookRAG system integrates and depends on MinerU 2.1.11, which was
released under the GNU Affero General Public License v3.0. The complete
runnable BookRAG system, when combined with MinerU, must therefore be used
and distributed in compliance with GNU AGPL v3.0.

Files containing or adapting MinerU example code are licensed only under
`AGPL-3.0-only` and are not offered under Apache-2.0.

## Excluded materials

Unless explicitly stated otherwise, the following materials are not covered
by the BookRAG software licenses:

- `Eval/`
- `datasets/`
- `assets/`
- `Scripts/`
- `config/`
- the BookRAG paper PDF
- other files outside `Core/`, except the explicitly licensed `main.py`

These materials may be subject to separate upstream licenses, dataset terms,
publication agreements, or copyright restrictions. No additional rights to
these materials are granted by the BookRAG software licenses.

## License texts

- GNU AGPL v3.0: [`LICENSE`](./LICENSE)
- Apache License 2.0: [`LICENSES/Apache-2.0.txt`](./LICENSES/Apache-2.0.txt)
- GNU AGPL v3.0 file-level copy:
  [`LICENSES/AGPL-3.0-only.txt`](./LICENSES/AGPL-3.0-only.txt)
- LightRAG MIT notice:
  [`LICENSES/LightRAG-MIT.txt`](./LICENSES/LightRAG-MIT.txt)

## Third-party components

Third-party software and adapted code remain subject to their respective
licenses. See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
