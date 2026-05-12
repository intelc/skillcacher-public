---
name: python-import-order
description: Organize Python imports per PEP 8 plus the isort default profile. Use when fixing import order in a Python file, when a linter complains about import grouping, or when reviewing diffs that touch imports.
---

When organizing Python imports, follow PEP 8 plus the isort default profile.
The aim is to produce an import block that is deterministic, easy to scan,
and stable under tooling — running isort and Black on the result must be a
no-op.

Group imports into three sections, separated by exactly one blank line:
standard library imports first, third-party imports second, first-party
(local project) imports third. Within each group, sort imports
alphabetically by module name, case-sensitively, with underscores treated
as ordinary characters. The standard library group includes everything
shipped with CPython, including modules whose names happen to look like
third-party packages (typing, dataclasses, pathlib are all standard
library). The third-party group is anything installed from PyPI or a
private index. The first-party group is anything inside the current
project's source tree, identified by isort's known_first_party setting or,
absent that, by the top-level package name matching the project name.

Within a single group, place plain `import` statements before any `from`
statements, then sort each subsection alphabetically. So
`import json` comes before `import os`, both come before `from collections
import OrderedDict`, and that comes before `from typing import Iterable`.
This is the isort "force_sort_within_sections=False" default; do not
change it on a per-file basis.

For `from X import Y` statements that import multiple names, place the
imported names inside parentheses on the next line, one name per line,
each name followed by a trailing comma, sorted alphabetically. The
trailing comma on the last entry is required so adding a new name later
produces a one-line diff rather than a two-line diff. Close the
parenthesis on its own line at the original indentation. Do not collapse
short multi-import statements onto one line even when they fit; the rule
is "always parens, always one name per line" so the shape never changes.

Conditional imports (inside `if TYPE_CHECKING:` blocks, inside
`try`/`except ImportError` fallbacks, inside functions for circular-import
breaking) live below the main import block and are not sorted alongside
unconditional imports. Mark them with a comment explaining why they are
conditional, so a future reader understands they were placed there
intentionally and not by mistake.

Do not import a module under an alias unless the alias is a project-wide
convention (numpy as np, pandas as pd, typing as t in some codebases). One-
off aliasing makes grep harder and obscures provenance.
