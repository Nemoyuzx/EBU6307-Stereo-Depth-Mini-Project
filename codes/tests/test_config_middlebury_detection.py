from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from config import resolve_middlebury_root


def _write_probe_scene(root: Path, scene_name: str = "artroom1") -> None:
    scene_dir = root / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "im0.png").write_bytes(b"")
    (scene_dir / "im1.png").write_bytes(b"")


class MiddleburyRootDetectionTests(unittest.TestCase):
    def test_school_shared_dataset_is_used_when_repo_data_has_no_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            repo_root = base_dir / "nemo-work"
            (repo_root / "data").mkdir(parents=True)
            shared_root = base_dir / "visual-computing-shared" / "MiddleburyDataset" / "data"
            _write_probe_scene(shared_root)

            resolved = resolve_middlebury_root(repo_root, "data")

            self.assertEqual(resolved, shared_root)

    def test_configured_dataset_wins_when_it_contains_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            repo_root = base_dir / "nemo-work"
            configured_root = repo_root / "data"
            shared_root = base_dir / "visual-computing-shared" / "MiddleburyDataset" / "data"
            _write_probe_scene(configured_root)
            _write_probe_scene(shared_root)

            resolved = resolve_middlebury_root(repo_root, "data")

            self.assertEqual(resolved, configured_root)


if __name__ == "__main__":
    unittest.main()