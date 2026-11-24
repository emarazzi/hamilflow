import sys
from pathlib import Path

# the tests import with --import-mode=importlib: make the shared synthetic-log helpers importable
sys.path.insert(0, str(Path(__file__).parent))
