import hashlib
import importlib
import os
import sys
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry


@contextmanager
def set_env_var(key, value):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


def load_user_module(module_path: str):
    module_path = os.path.abspath(module_path)
    module_parent, module_name = os.path.split(module_path)
    if module_name not in sys.modules:
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)


def download(
    url: str,
    save_path: os.PathLike,
    keep_name: bool = False,
    skip_if_exists: bool = False,
    timeout: tuple[float, float] = (5.0, 60.0),
    retries: int = 3,
    chunk_size: int = 8192,
    user_agent: str = "FlexRAG Downloader/1.0",
    expected_size: int | None = None,
    hash_type: str | None = None,  # e.g. "md5", "sha256"
    hash_value: str | None = None,
    proxies: dict[str, str] | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Download a file from a URL to a local path.

    :param url: The URL of the file to download.
    :type url: str
    :param save_path: The local path to save the downloaded file.
    :type save_path: os.PathLike
    :param keep_name: If True, keep the original file name when saving.
    :type keep_name: bool
    :param skip_if_exists: Skip downloading if the file already exists.
    :type skip_if_exists: bool
    :param timeout: A tuple specifying the connection and read timeout in seconds.
        Defaults to (5.0, 60.0).
    :type timeout: tuple[float, float]
    :param retries: Number of retries for failed downloads.
        Default is 3.
    :type retries: int
    :param chunk_size: The chunk size for downloading the file.
        Defaults to 8192 bytes.
    :type chunk_size: int
    :param user_agent: The User-Agent header to use for the download request.
        Default is "FlexRAG Downloader/1.0".
    :type user_agent: str
    :param expected_size: If provided, verify the downloaded file size.
    :type expected_size: int | None
    :param hash_type: The hash type for verification (e.g., "md5", "sha256").
    :type hash_type: str | None
    :param hash_value: The expected hash value for verification.
        If provided, verify the downloaded file hash.
    :type hash_value: str | None
    :param proxies: A dictionary of proxy settings for requests.
    :type proxies: dict[str, str] | None
    :param show_progress: Whether to show a progress bar during download.
    :type show_progress: bool
    :return: The path to the downloaded file.
    :rtype: Path
    """

    # determine final saving path
    url_filename = Path(url).name
    save_path = Path(save_path)
    if keep_name:
        final_path = save_path / url_filename
    else:
        final_path = save_path

    # create directory if not exists
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # skip if exists
    if skip_if_exists and final_path.is_file():
        return final_path

    session = requests.Session()
    retry_cfg = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("http://", HTTPAdapter(max_retries=retry_cfg))
    session.mount("https://", HTTPAdapter(max_retries=retry_cfg))

    headers = {"User-Agent": user_agent}
    with session.get(
        url, stream=True, timeout=timeout, headers=headers, proxies=proxies
    ) as resp:
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if expected_size is not None and content_length is not None:
            if int(content_length) != expected_size:
                raise ValueError(
                    f"Size mismatch: header {content_length} != expected {expected_size}"
                )

        hasher = hashlib.new(hash_type) if hash_type else None

        # download to temporary file
        temp_path = final_path.with_suffix(".download_tmp")
        with (
            open(temp_path, "wb") as f,
            tqdm(
                total=int(content_length) if content_length is not None else None,
                unit="B",
                unit_scale=True,
                desc="Downloading",
                disable=not show_progress,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if hasher:
                    hasher.update(chunk)
                pbar.update(len(chunk))

        # verify size
        if expected_size is not None:
            actual_size = temp_path.stat().st_size
            if actual_size != expected_size:
                temp_path.unlink()
                raise ValueError(
                    f"Written size {actual_size} != expected {expected_size}"
                )

        # verify hash
        if hasher:
            digest = hasher.hexdigest()
            if digest.lower() != hash_value.lower():
                temp_path.unlink()
                raise ValueError(f"Hash mismatch: {digest} != {hash_value}")

        temp_path.rename(final_path)

    return final_path


def download_and_extract(
    url: str,
    save_dir: os.PathLike,
    keep_original: bool = False,
    skip_if_exists: bool = False,
    timeout: tuple[float, float] = (5.0, 60.0),
    retries: int = 3,
    chunk_size: int = 8192,
    user_agent: str = "FlexRAG Downloader/1.0",
    expected_size: int | None = None,
    hash_type: str | None = None,  # e.g. "md5", "sha256"
    hash_value: str | None = None,
    proxies: dict[str, str] | None = None,
    show_progress: bool = True,
) -> Path:
    """Download and extract a compressed file from a URL to a local directory.

    :param url: The URL of the file to download.
    :type url: str
    :param save_dir: The local directory to save the extracted files.
    :type save_dir: os.PathLike
    :param keep_original: If True, keep the original compressed file after extraction.
    :type keep_original: bool
    :param skip_if_exists: Skip downloading if the file already exists.
    :type skip_if_exists: bool
    :param timeout: A tuple specifying the connection and read timeout in seconds.
        Defaults to (5.0, 60.0).
    :type timeout: tuple[float, float]
    :param retries: Number of retries for failed downloads.
        Default is 3.
    :type retries: int
    :param chunk_size: The chunk size for downloading the file.
        Defaults to 8192 bytes.
    :type chunk_size: int
    :param user_agent: The User-Agent header to use for the download request.
        Default is "FlexRAG Downloader/1.0".
    :type user_agent: str
    :param expected_size: If provided, verify the downloaded file size.
    :type expected_size: int | None
    :param hash_type: The hash type for verification (e.g., "md5", "sha256").
    :type hash_type: str | None
    :param hash_value: The expected hash value for verification.
        If provided, verify the downloaded file hash.
    :type hash_value: str | None
    :param proxies: A dictionary of proxy settings for requests.
    :type proxies: dict[str, str] | None
    :param show_progress: Whether to show a progress bar during download.
    :type show_progress: bool
    :return: The path to the downloaded directory.
    :rtype: Path
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / Path(url).name
    download(
        url=url,
        save_path=filename,
        keep_name=False,
        skip_if_exists=skip_if_exists,
        timeout=timeout,
        retries=retries,
        chunk_size=chunk_size,
        user_agent=user_agent,
        expected_size=expected_size,
        hash_type=hash_type,
        hash_value=hash_value,
        proxies=proxies,
        show_progress=show_progress,
    )

    if tarfile.is_tarfile(filename):
        with tarfile.open(filename, "r:*") as tar:
            tar.extractall(path=save_dir)
    elif zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=save_dir)
    else:
        raise ValueError(f"Unsupported archive format for file: {filename}")

    if not keep_original:
        filename.unlink()
    return save_dir
