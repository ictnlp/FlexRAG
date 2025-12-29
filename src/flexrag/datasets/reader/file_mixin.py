import bz2
import gzip
import io
import lzma
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import lz4.frame
except ImportError:
    lz4 = None


class FileReaderMixin:
    """Provide _open_file that transparently opens plain or compressed files."""

    @contextmanager
    def _open_file(self, path, mode="r", encoding=None, errors=None):
        path = Path(path)
        with ExitStack() as stack:
            suffix = path.suffix.lower()
            if suffix == ".gz":
                f = stack.enter_context(
                    gzip.open(
                        path,
                        mode="rt" if "b" not in mode else mode,
                        encoding=encoding,
                        errors=errors,
                    )
                )
            elif suffix == ".bz2":
                f = stack.enter_context(
                    bz2.open(
                        path,
                        mode="rt" if "b" not in mode else mode,
                        encoding=encoding,
                        errors=errors,
                    )
                )
            elif suffix in (".xz", ".lzma"):
                f = stack.enter_context(
                    lzma.open(
                        path,
                        mode="rt" if "b" not in mode else mode,
                        encoding=encoding,
                        errors=errors,
                    )
                )
            elif suffix == ".zip":
                z = stack.enter_context(zipfile.ZipFile(path))
                names = z.namelist()
                if not names:
                    raise ValueError(f"Empty zip archive: {path}")
                # pick first file in archive
                b = stack.enter_context(z.open(names[0], "r"))
                f = io.TextIOWrapper(b, encoding=encoding, errors=errors)
            elif suffix in (".zstd", ".zst"):
                if zstd is None:
                    raise ImportError(
                        "zstandard module is not installed. "
                        "Please install it via 'pip install zstandard'."
                    )
                f = stack.enter_context(
                    zstd.open(
                        path,
                        mode="rt" if "b" not in mode else mode,
                        encoding=encoding,
                        errors=errors,
                    )
                )
            elif suffix == ".lz4":
                if lz4 is None:
                    raise ImportError(
                        "lz4 module is not installed. "
                        "Please install it via 'pip install lz4'."
                    )
                f = stack.enter_context(
                    lz4.frame.open(
                        path,
                        mode="rt" if "b" not in mode else mode,
                        encoding=encoding,
                        errors=errors,
                    )
                )
            else:
                f = stack.enter_context(
                    open(path, mode, encoding=encoding, errors=errors)
                )
            yield f
