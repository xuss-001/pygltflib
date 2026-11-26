"""Compatibility module so ``from pygltflib.merge import merge_gltfs`` works.

The implementation lives in :mod:`pygltflib.merger`.  This module is also
callable, so ``pygltflib.merge(...)`` works whether the name resolves to the
package-level function or to this imported submodule.
"""

import sys
import types

from .merger import *  # noqa: F401,F403
from .merger import (  # noqa: F401
    InvalidInputError,
    InvalidGltfReferenceError,
    InvalidReferenceError,
    MergeEntry,
    MergeError,
    MergeException,
    MergeInput,
    Merger,
    UnsupportedFeatureError,
    UnsupportedGltfError,
    UnsupportedGltfFeatureError,
    GltfMergeError,
    combine_gltf,
    combine_gltfs,
    merge,
    merge_files,
    merge_gltf,
    merge_gltfs,
    merge_models,
    merge_scenes,
)


class _CallableMergeModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return merge_gltfs(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableMergeModule
