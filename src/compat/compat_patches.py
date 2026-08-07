"""
Lightweight compatibility patches for old dependencies.

Import this once near program startup so mujoco_py, gym, design_bench, and
other older libraries do not fail on deprecated aliases in Python 3.10+ or
newer numpy versions.
"""

import numpy as np
import collections
import collections.abc


if "bool" not in np.__dict__:
    np.bool = np.bool_
if "int" not in np.__dict__:
    np.int = int
if "float" not in np.__dict__:
    np.float = float


if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
