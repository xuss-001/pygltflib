"""Compatibility shim: ``from pygltflib.merge_gltf import merge_gltf``."""

import sys
import types

from .merger import *  # noqa: F401,F403
from .merger import merge_gltf, merge_gltfs  # noqa: F401


class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return merge_gltfs(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
