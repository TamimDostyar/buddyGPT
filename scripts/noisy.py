"""Shim re-exporting scrub from adapters.base."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from adapters.base import scrub, URL_RE, SEPARATOR_RE

__all__ = ["scrub", "URL_RE", "SEPARATOR_RE"]
