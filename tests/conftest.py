"""Make the repo root importable so `finprint` and `app` resolve when pytest is
run from anywhere, and expose the signal helpers to every test module."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
