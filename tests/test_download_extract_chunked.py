import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.download_extract_chunked import (
    RollingPartReader,
    extract_stream,
    load_include_names,
    member_include_key,
)


def _add_tar_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


class TestDownloadExtractChunked(unittest.TestCase):
    def test_include_list_normalizes_chunked_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "include.txt"
            include.write_text(
                "\n".join(
                    [
                        "# comments are ignored",
                        "chunked_videos/298.mp4",
                        "./chunked_videos/471.mp4",
                        "472.mp4",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_include_names(str(include)), {"298.mp4", "471.mp4", "472.mp4"})
            self.assertEqual(member_include_key("chunked_videos/298.mp4"), "298.mp4")

    def test_extract_stream_filters_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "chunked_videos.tar.partaa"
            out_dir = root / "chunked_videos"

            with tarfile.open(part, "w") as tar:
                _add_tar_member(tar, "chunked_videos/keep.mp4", b"keep")
                _add_tar_member(tar, "chunked_videos/drop.mp4", b"drop")

            reader = RollingPartReader([part], total_bytes=part.stat().st_size)
            try:
                count = extract_stream(reader, out_dir, include_names={"keep.mp4"})
            finally:
                reader.close()

            self.assertEqual(count, 1)
            self.assertEqual((out_dir / "keep.mp4").read_bytes(), b"keep")
            self.assertFalse((out_dir / "drop.mp4").exists())
            self.assertFalse(part.exists())


if __name__ == "__main__":
    unittest.main()
