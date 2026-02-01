"""Configure sys.path for imports from ace/ and ShinkaEvolve/."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

# Add ace/ and ShinkaEvolve/ to path so we can import them
sys.path.insert(0, str(repo_root / "ace"))
sys.path.insert(0, str(repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(repo_root / "src"))
