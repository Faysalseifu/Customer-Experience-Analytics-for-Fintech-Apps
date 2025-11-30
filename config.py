"""
Compatibility shim: expose `config` at project root by re-exporting
from `Scripts.config`. This keeps existing import patterns working
without modifying `Scripts/preprocessing.py`.
"""
from Scripts.config import *  # noqa: F401,F403
