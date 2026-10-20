from __future__ import annotations

import site
from pathlib import Path


# Enables local verification in this prepared workspace; normal installations use pyproject.toml.
local_packages = Path(__file__).parents[1] / ".packages"
if local_packages.exists():
    site.addsitedir(str(local_packages))
