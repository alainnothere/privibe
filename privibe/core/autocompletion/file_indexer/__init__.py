from __future__ import annotations

# The persistent index/store/watcher were removed in favour of stateless,
# read-once-per-@-session enumeration (see core/autocompletion/file_enumerator.py).
# This package now only hosts ignore_rules, which is shared with project_tree and
# the local-config walk.
