import bz2
import gzip
import io
import lzma
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path


class FileReaderMixin:
    """Provide _open_file that transparently opens plain or compressed files."""

    @contextmanager
    def _open_file(self, path, mode="r", encoding=None, errors=None):
        path = Path(path)
        stack = ExitStack()
        try:
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
            else:
                f = stack.enter_context(
                    open(path, mode, encoding=encoding, errors=errors)
                )
            yield f
        finally:
            stack.close()
