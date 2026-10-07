from __future__ import annotations

import collections.abc
import sys
from contextlib import contextmanager
from types import ModuleType


@contextmanager
def legacy_timm_import():
    installed = "torch._six" not in sys.modules
    if installed:
        module = ModuleType("torch._six")
        module.container_abcs = collections.abc
        sys.modules["torch._six"] = module
    try:
        yield
    finally:
        if installed:
            del sys.modules["torch._six"]
