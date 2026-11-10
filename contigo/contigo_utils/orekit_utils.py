"""Orekit start-up: helper jar, Orekit data and the JVM.

- :func:`helper_jar_path`: the contigo helper jar shipped as package data
  (``contigo/java/orekit_utils-1.0.0.jar``).
- :func:`resolve_orekit_data`: the ``orekit_data_<sha8>.zip`` to use, kept in
  ``config.cache_dir("orekit")``. Runs before the JVM starts, so a failure
  leaves no JVM behind. Offline (``CONTIGO_OFFLINE=1``) or when GitLab cannot
  be reached, the newest local zip is used (cache first, then the legacy
  ``contigo/data``).
- :func:`start_orekit`: resolves the data, starts the JVM and loads the data.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import warnings
import zipfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional, Union

import jdk4py
import jpype
import orekit_jpype as orekit
import requests

import contigo.config as config
import contigo.contigo_utils.utils as utils

logger = logging.getLogger(__name__)

JAR_NAME = "orekit_utils-1.0.0.jar"
OREKIT_DATA_PROJECT_ID = 18
OREKIT_DATA_URL = (
    "https://gitlab.orekit.org/orekit/orekit-data/-/archive/{sha}/"
    "orekit-data-{sha}.zip"
)
DATA_GLOB = "orekit_data_*.zip"

# Keeps the as_file() context (a temporary copy for zipped installs) open for
# the life of the process; closed at interpreter exit.
_RESOURCES = contextlib.ExitStack()
atexit.register(_RESOURCES.close)
_JAR_CACHE: Optional[Path] = None


class OrekitDataNotFoundError(FileNotFoundError):
    """No usable ``orekit_data_*.zip`` is available locally or for download."""


# ---------------------------------------------------------------------------
# Helper jar
# ---------------------------------------------------------------------------
def helper_jar_path() -> Path:
    """Path of the contigo Orekit helper jar.

    Looked up as package data (``contigo/java/``) first. A jar built into the
    old location ``java_src/target/`` of a source checkout is still accepted
    for one release, with a ``FutureWarning`` (shown by default).

    Raises
    ------
    FileNotFoundError
        If neither location holds the jar.
    """
    global _JAR_CACHE
    if _JAR_CACHE is not None and _JAR_CACHE.is_file():
        return _JAR_CACHE

    res = files("contigo") / "java" / JAR_NAME
    if res.is_file():
        _JAR_CACHE = Path(_RESOURCES.enter_context(as_file(res))).resolve()
        return _JAR_CACHE

    legacy = Path(__file__).resolve().parents[2] / "java_src" / "target" / JAR_NAME
    if legacy.is_file():
        warnings.warn(
            f"Using the Orekit helper jar from the old location {legacy}. "
            "It now belongs in contigo/java/ (`cd java_src && mvn package` "
            "writes it there); the old location will stop working in the "
            "next release.",
            FutureWarning,
            stacklevel=2,
        )
        return legacy

    raise FileNotFoundError(
        f"Orekit helper jar {JAR_NAME} not found in the contigo package "
        f"(contigo/java/) or in {legacy.parent}. Reinstall contigo, or build it "
        "from a source checkout with `cd java_src && mvn package`."
    )


# ---------------------------------------------------------------------------
# Orekit data
# ---------------------------------------------------------------------------
def get_gitlab_sha(project_id=OREKIT_DATA_PROJECT_ID, branch="main", timeout=10):
    """SHA of the latest commit on ``branch`` of a gitlab.orekit.org project.

    Raises
    ------
    contigo.config.OfflineError
        If ``CONTIGO_OFFLINE`` is set (no request is made).
    requests.HTTPError
        On a 404 (wrong project ID or branch) or another HTTP error status.
    requests.RequestException
        On connection errors and timeouts.
    ValueError
        If a 200 response has no string ``id`` (not JSON, not a dict, captive
        portal, API change).
    """
    # Note: the branch was renamed from 'master' to 'main'
    url_base = "https://gitlab.orekit.org/api/v4/projects/"
    url = f"{url_base}{project_id}/repository/commits/{branch}"
    utils._require_online(url)
    response = requests.get(url, timeout=timeout)

    if response.status_code == 404:
        raise requests.HTTPError(
            f"404 from {url}: check the project ID ({project_id}) and branch "
            f"name ({branch!r}).",
            response=response,
        )

    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as err:  # includes JSONDecodeError
        raise ValueError(f"{url} did not return JSON: {err}") from err
    sha = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"{url} returned no commit id: {str(payload)[:200]!r}")
    return sha


def _zips_in(d: Path) -> List[Path]:
    """``orekit_data_*.zip`` files in ``d``, newest mtime first."""
    if not d.is_dir():
        return []
    found = [p for p in d.glob(DATA_GLOB) if p.is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _local_data_zips() -> List[Path]:
    """Local Orekit data zips in order of preference.

    Cache zips (newest mtime first) always come before the legacy
    ``DATA_DIR`` zips, whose mtime a git checkout or pull refreshes.
    """
    return _zips_in(config.cache_dir("orekit", create=False)) + _zips_in(
        Path(config.DATA_DIR)
    )


def _not_found_message(reason: str, offline: bool) -> str:
    where = (
        f"No local Orekit data zip ({DATA_GLOB}) was found in "
        f"{config.cache_dir('orekit', create=False)} or {config.DATA_DIR}."
    )
    place = (
        "place an orekit-data zip named orekit_data_<sha8>.zip in "
        f"{config.cache_dir('orekit', create=False)}"
    )
    if offline:
        fix = (
            "Unset CONTIGO_OFFLINE once (with network access) so the data can "
            f"be downloaded, or {place}."
        )
    else:
        fix = f"Check network access to gitlab.orekit.org, or {place}."
    return f"{reason} {where} {fix}"


def _newest_local_or_raise(
    reason: str, cause: Optional[BaseException] = None, offline: bool = False
) -> Path:
    local = _local_data_zips()
    if local:
        return local[0]
    raise OrekitDataNotFoundError(_not_found_message(reason, offline)) from cause


def _fallback(reason: str, err: BaseException) -> Path:
    """Newest local zip with a warning, or raise chained to ``err``."""
    path = _newest_local_or_raise(reason, err)
    warnings.warn(
        f"{reason} Using the local file {path}.", RuntimeWarning, stacklevel=3
    )
    return path


def _validate_zip(path: Path, expected_size: int) -> None:
    """Raise ValueError unless ``path`` is a complete, readable zip."""
    if expected_size and path.stat().st_size != expected_size:
        raise ValueError(
            f"size {path.stat().st_size} does not match Content-Length "
            f"{expected_size}"
        )
    if not zipfile.is_zipfile(path):
        raise ValueError("not a zip file")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise ValueError(f"corrupt member {bad!r}")


def resolve_orekit_data() -> Path:
    """Return the Orekit data zip to load, downloading it if needed.

    Does not start the JVM. Never writes into ``config.DATA_DIR``.

    Local zips are preferred in this order: ``config.cache_dir("orekit")``
    (newest mtime first), then the legacy ``config.DATA_DIR``.

    - Offline (``CONTIGO_OFFLINE``): the first local zip.
    - Online: the latest orekit-data commit SHA is queried. If
      ``orekit_data_<sha8>.zip`` is in the cache or ``DATA_DIR`` it is used;
      otherwise it is downloaded to a per-process ``.part`` file, checked to
      be a complete zip and moved into place, and older zips in the cache are
      deleted. If the query, the download or the check fails, the first local
      zip is used with a warning.

    Raises
    ------
    OrekitDataNotFoundError
        If no zip can be downloaded and none exists locally.
    """
    if config.is_offline():
        path = _newest_local_or_raise("CONTIGO_OFFLINE is set.", offline=True)
        logger.info("Offline: using local Orekit data %s", path)
        return path

    try:
        sha = get_gitlab_sha(project_id=OREKIT_DATA_PROJECT_ID, branch="main")
    except (requests.RequestException, ValueError, KeyError, TypeError) as err:
        return _fallback(
            f"Could not query the latest Orekit data version ({err}).", err
        )

    name = f"orekit_data_{sha[:8]}.zip"
    cache = config.cache_dir("orekit")
    target = cache / name
    for cand in (target, Path(config.DATA_DIR) / name):
        if cand.is_file():
            return cand

    part = cache / f"{name}.{os.getpid()}.part"
    url = OREKIT_DATA_URL.format(sha=sha)
    logger.info("Downloading Orekit data %s to %s", url, target)
    try:
        expected = utils.dl_file(url, str(part))
        _validate_zip(part, expected if isinstance(expected, int) else 0)
        os.replace(part, target)
    except Exception as err:
        with contextlib.suppress(OSError):
            part.unlink(missing_ok=True)
        return _fallback(f"Downloading the Orekit data from {url} failed ({err}).", err)

    for old in cache.glob(DATA_GLOB):
        if old != target:
            logger.info("Removing old Orekit data file %s", old)
            with contextlib.suppress(OSError):
                old.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------------------
# JVM
# ---------------------------------------------------------------------------
def start_orekit(
    vmargs: Union[str, None] = None,
    additional_classpaths: Union[List, None] = None,
    jvmpath: Optional[Union[str, os.PathLike]] = None,
):
    """Start the Orekit JVM with the contigo helper jar and load Orekit data.

    The data zip is resolved (see :func:`resolve_orekit_data`) before the JVM
    starts, so a missing zip raises without leaving a JVM running. Does
    nothing if the JVM is already running (with a warning if contigo did not
    start it, since a JVM classpath cannot be changed after start).

    ``vmargs`` is accepted for backward compatibility and is unused.
    """
    if jpype.isJVMStarted():
        if not config.state.get("orekit_loaded"):
            logger.warning(
                "The JVM was started outside contigo; the contigo helper jar "
                "and Orekit data may not be configured (a JVM classpath cannot "
                "be changed after start)."
            )
        return

    data_path = resolve_orekit_data()
    if additional_classpaths is None:
        additional_classpaths = [str(helper_jar_path())]

    logger.info("Starting Orekit JVM")
    # use the jdk4py JDK, which is compatible with orekit, for every user
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    orekit.initVM(jvmpath=jvmpath, additional_classpaths=additional_classpaths)

    from orekit_jpype.pyhelpers import setup_orekit_data

    logger.info("Loading Orekit data from %s", data_path)
    setup_orekit_data(filenames=str(data_path), from_pip_library=False)

    config.state["orekit_loaded"] = True
