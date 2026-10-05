"""Pytest root config: makes `import module_05_ppe` work from the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
