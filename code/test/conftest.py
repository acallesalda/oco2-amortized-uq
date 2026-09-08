"""conftest.py
Put `code/` on `sys.path`, the way every script in this repository does.

The project is not installed as a package: `code/` holds top-level modules (`data`,
`evals`, `modeling`, `utils`) and every entry point inserts that directory itself. The
tests do the same, so `pytest code/test` works from the repository root with nothing
installed and no `PYTHONPATH` set.
"""

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
