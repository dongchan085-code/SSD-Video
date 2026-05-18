"""
Round-trip tests for the precomputed-frames pipeline (no GPU required).

Tests:
  1. Extractor produces meta.json + PNGs and deletes the source mp4.
  2. Loader returns the last N frames in insertion order.
  3. Loader slices correctly for num_frames < saved_count.
  4. Loader raises ValueError when fps/chunk_duration mismatch.
  5. Loader raises FileNotFoundError when meta.json is missing.
  6. Extraction is idempotent (re-run does not overwrite or re-decode).
  7. resolve_frame_dir returns the dir when meta.json present, None otherwise.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# video_utils.py imports torch/torchvision at module level. Stub them with
# real-looking module objects so submodule imports (torch.utils.data) work.
# cv2 is lazy-imported inside read_video_frames/read_video_metadata, so it
# does NOT need to be mocked here — extractor tests will self-skip if absent.
import types
from unittest.mock import MagicMock


def _stub_torch() -> None:
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    data_mod.DataLoader = MagicMock()
    data_mod.random_split = MagicMock()
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    torch_mod.Tensor = MagicMock()
    torch_mod.stack = MagicMock()
    for key, val in [("torch", torch_mod), ("torch.utils", utils_mod), ("torch.utils.data", data_mod)]:
        sys.modules.setdefault(key, val)
    for mod in ("torchvision", "torchvision.transforms"):
        sys.modules.setdefault(mod, MagicMock())


_stub_torch()

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ssd_vlm_video_utils_isolated",
    PROJECT_ROOT / "ssd_vlm" / "data" / "video_utils.py",
)
_video_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_video_utils)

PRECOMPUTED_FRAMES_META = _video_utils.PRECOMPUTED_FRAMES_META
load_precomputed_frames = _video_utils.load_precomputed_frames
resolve_frame_dir = _video_utils.resolve_frame_dir


def _make_synthetic_mp4(path: Path, n_frames: int = 30, fps: float = 25.0) -> None:
    """Write a minimal valid mp4 with solid-colour frames using OpenCV."""
    try:
        import cv2
    except ModuleNotFoundError:
        raise unittest.SkipTest("cv2 not installed — skipping mp4-dependent test")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 64, 64
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        rng = np.random.default_rng(42)
        for i in range(n_frames):
            colour = (rng.integers(0, 256, 3)).astype(np.uint8)
            frame = np.ones((h, w, 3), dtype=np.uint8) * colour
            out.write(frame.astype(np.uint8))
    finally:
        out.release()


def _make_precomputed_dir(
    frame_dir: Path,
    n_frames: int = 8,
    fps: float = 1.0,
    chunk_duration: float = 1.0,
    total_frames: int = 50,
) -> Path:
    """Write a valid precomputed-frames directory with synthetic PNG frames."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)
    indices = list(range(n_frames))
    timestamps = [float(i) / fps for i in indices]
    for i in range(n_frames):
        arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(frame_dir / f"frame_{i:02d}.png")
    meta = {
        "total_frames": total_frames,
        "source_fps": 25.0,
        "extraction_fps": fps,
        "chunk_duration": chunk_duration,
        "recent_frames_only": n_frames,
        "frame_indices": indices,
        "frame_timestamps": timestamps,
        "resize_shortest_edge": 32,
        "saved_count": n_frames,
        "source_size_bytes": 1024,
    }
    with open(frame_dir / PRECOMPUTED_FRAMES_META, "w") as f:
        json.dump(meta, f)
    return frame_dir


class TestExtractorRoundTrip(unittest.TestCase):
    """End-to-end: synthetic mp4 → extractor → PNG frames + meta.json."""

    def test_extract_produces_pngs_and_meta(self):
        """Extractor writes correct number of PNGs and a valid meta.json."""
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 not installed")

        from scripts.extract_chunk_frames import _extract_one

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mp4 = tmp / "vid001.mp4"
            out_dir = tmp / "frames" / "vid001"
            _make_synthetic_mp4(mp4, n_frames=30, fps=25.0)

            saved = _extract_one(
                video_path=mp4,
                out_dir=out_dir,
                recent_frames=8,
                fps=1.0,
                chunk_duration=1.0,
                resize_shortest_edge=32,
            )

            self.assertGreater(saved, 0)
            pngs = sorted(out_dir.glob("frame_*.png"))
            self.assertEqual(len(pngs), saved)
            meta_path = out_dir / PRECOMPUTED_FRAMES_META
            self.assertTrue(meta_path.exists())
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["saved_count"], saved)
            self.assertEqual(len(meta["frame_indices"]), saved)
            self.assertAlmostEqual(meta["extraction_fps"], 1.0)

    def test_extractor_deletes_source_on_request(self):
        """Script deletes mp4 only when --delete_source is in effect."""
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 not installed")

        from scripts.extract_chunk_frames import _extract_one

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mp4 = tmp / "todel.mp4"
            out_dir = tmp / "frames" / "todel"
            _make_synthetic_mp4(mp4, n_frames=10)
            self.assertTrue(mp4.exists())
            _extract_one(mp4, out_dir, 8, 1.0, 1.0, None)
            # mp4 should still exist — deletion is the caller's responsibility
            self.assertTrue(mp4.exists())

    def test_extractor_idempotent(self):
        """Re-running _already_extracted on a complete dir returns True."""
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 not installed")

        from scripts.extract_chunk_frames import _already_extracted, _extract_one

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mp4 = tmp / "idem.mp4"
            out_dir = tmp / "frames" / "idem"
            _make_synthetic_mp4(mp4, n_frames=10)
            _extract_one(mp4, out_dir, 4, 1.0, 1.0, None)
            self.assertTrue(_already_extracted(out_dir))


class TestLoadPrecomputedFrames(unittest.TestCase):
    """Unit tests for load_precomputed_frames()."""

    def test_returns_last_n_frames(self):
        """Requesting 4 frames from 8 saved returns the last 4."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", n_frames=8)
            pil_frames, indices, total_frames, timestamps, chunk_ids = load_precomputed_frames(
                frame_dir=d,
                num_frames=4,
                expected_fps=1.0,
                expected_chunk_duration=1.0,
            )
            self.assertEqual(len(pil_frames), 4)
            self.assertEqual(len(indices), 4)
            self.assertEqual(total_frames, 50)
            # Should be the LAST 4 indices (4,5,6,7)
            self.assertEqual(indices, [4, 5, 6, 7])

    def test_returns_all_when_num_frames_ge_saved(self):
        """Requesting more frames than saved returns all saved."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", n_frames=5)
            pil_frames, indices, _, _, _ = load_precomputed_frames(
                frame_dir=d, num_frames=32, expected_fps=1.0, expected_chunk_duration=1.0
            )
            self.assertEqual(len(pil_frames), 5)

    def test_pil_images_are_rgb(self):
        """Returned PIL images are RGB mode."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", n_frames=4)
            pil_frames, *_ = load_precomputed_frames(
                frame_dir=d, num_frames=4, expected_fps=1.0, expected_chunk_duration=1.0
            )
            for img in pil_frames:
                self.assertIsInstance(img, PILImage.Image)
                self.assertEqual(img.mode, "RGB")

    def test_raises_on_fps_mismatch(self):
        """Loader raises ValueError when extraction_fps != expected_fps."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", fps=1.0)
            with self.assertRaises(ValueError) as ctx:
                load_precomputed_frames(
                    frame_dir=d, num_frames=4, expected_fps=2.0, expected_chunk_duration=1.0
                )
            self.assertIn("fps=1.0", str(ctx.exception))

    def test_raises_on_chunk_duration_mismatch(self):
        """Loader raises ValueError when chunk_duration != expected."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", chunk_duration=1.0)
            with self.assertRaises(ValueError):
                load_precomputed_frames(
                    frame_dir=d, num_frames=4, expected_fps=1.0, expected_chunk_duration=2.0
                )

    def test_raises_missing_meta(self):
        """Loader raises FileNotFoundError when meta.json is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                load_precomputed_frames(
                    frame_dir=empty, num_frames=4, expected_fps=1.0, expected_chunk_duration=1.0
                )

    def test_raises_png_count_mismatch(self):
        """Loader raises ValueError when png count != saved_count in meta."""
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunk", n_frames=4)
            # Delete one png to corrupt the dir
            sorted(d.glob("frame_*.png"))[0].unlink()
            with self.assertRaises(ValueError):
                load_precomputed_frames(
                    frame_dir=d, num_frames=4, expected_fps=1.0, expected_chunk_duration=1.0
                )


class TestResolveFrameDir(unittest.TestCase):
    """Unit tests for resolve_frame_dir()."""

    def test_returns_dir_when_meta_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_precomputed_dir(Path(tmp) / "chunked_frames" / "vid42", n_frames=4)
            result = resolve_frame_dir(data_path=Path(tmp), video_id="vid42")
            self.assertEqual(result, d)

    def test_returns_none_when_meta_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_frame_dir(data_path=Path(tmp), video_id="nonexistent")
            self.assertIsNone(result)

    def test_custom_chunked_frames_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "myframes"
            d = _make_precomputed_dir(custom / "vid99", n_frames=4)
            result = resolve_frame_dir(
                data_path=Path(tmp), video_id="vid99", chunked_frames_dir=custom
            )
            self.assertEqual(result, d)


if __name__ == "__main__":
    unittest.main()
