"""Type definitions and global variables."""

from typing import Any, TypeAlias

STRING_CLASSES = (str, bytes)

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1
Key: TypeAlias = str | tuple[str, ...]
Value: TypeAlias = Any
Tree: TypeAlias = dict[Key, Value]
