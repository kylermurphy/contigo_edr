"""Utilites for the contigo module

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import bz2
import functools
import gzip
import io
import logging
import lzma
import pathlib
import shutil
import zipfile
from collections.abc import Iterator
from datetime import datetime

import pandas as pd
import requests
from dateutil import tz
from tqdm import tqdm

import contigo.config as config

logger = logging.getLogger(__name__)


def _require_online(url):
    """Raise ``config.OfflineError`` if ``CONTIGO_OFFLINE`` is set."""
    if config.is_offline():
        raise config.OfflineError(f"CONTIGO_OFFLINE is set; refusing to download {url}")


def dl_file(url, filename, timeout=(10, 60)):
    """
    Download file from url to filename

    Parameters
    ----------
    url : url to file.
    filename : filename to save as.
    timeout : (connect, read) timeout in seconds passed to ``requests``, so a
        stalled connection raises instead of hanging.

    Returns
    -------
    int
        The expected size in bytes from ``Content-Length``, or 0 if the server
        sent none or the body is content-encoded (then the size on disk can
        differ from the header).

    Raises
    ------
    for
        DESCRIPTION.
    RuntimeError
        DESCRIPTION.
    contigo.config.OfflineError
        If ``CONTIGO_OFFLINE`` is set (no request is made).

    """
    _require_online(url)
    r = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    # if requests couldn't get file size drop encoding to get it
    if file_size == 0:
        rs = requests.head(url, headers={"Accept-Encoding": None}, timeout=timeout)
        file_size = int(rs.headers.get("Content-Length", 0))

    fpath = pathlib.Path(filename).expanduser().resolve()
    fpath.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    post = f"Downloading: {url} to {filename}"
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(
        r.raw, "read", total=file_size, desc=desc, postfix=post, position=0, leave=True
    ) as r_raw:
        with fpath.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    encoding = str(r.headers.get("Content-Encoding", "identity")).lower()
    return file_size if encoding in ("", "identity") else 0


def wf_mtime(url):
    """
    Retrieves the last modified date of a file on the web.

    Args:
        url (str): The URL of the file.

    Returns:
        datetime or None: The last modified date as a datetime object,
                          or None if the header is not found, an error occurs,
                          or ``CONTIGO_OFFLINE`` is set (no request is made).
    """
    if config.is_offline():
        return None
    try:
        response = requests.head(url)  # Use HEAD request for efficiency
        response.raise_for_status()  # Raise an exception for bad status codes

        last_modified_header = response.headers.get("Last-Modified")

        if last_modified_header:
            # Parse the date string from the header
            # Example format: 'Wed, 21 Oct 2015 07:28:00 GMT'
            return datetime.strptime(
                last_modified_header, "%a, %d %b %Y %H:%M:%S %Z"
            ).replace(tzinfo=tz.gettz("GMT"))
        else:
            return None
    except requests.exceptions.RequestException as e:
        logger.warning("Could not get the modification time of %s: %s", url, e)
        return None


#: SP3 "bad or absent value" sentinel (clock and, in practice, other fields)
_SP3_BAD = 999999.0

#: Magic bytes of compressed formats; SP3 text is ASCII, so anything else is
#: read as plain text.
_MAGIC_GZIP = b"\x1f\x8b"
_MAGIC_COMPRESS = b"\x1f\x9d"
_MAGIC_BZ2 = b"BZh"
_MAGIC_XZ = b"\xfd7zXZ\x00"


def _sp3_lines(fn: pathlib.Path) -> Iterator[str]:
    """Yield the text lines of a plain, gzip, bzip2, xz or zip SP3 file.

    The compression is detected from the file's magic bytes, not its name.
    Unix ``compress`` (``.Z``, LZW) files cannot be read by the standard
    library, so they raise a clear error instead of being decoded as garbage.
    Text is decoded as ASCII with ``errors='replace'``: only fixed-width
    numeric columns are parsed, so stray bytes in comments are harmless.
    """
    with open(fn, "rb") as fb:
        magic = fb.read(6)

    def text(binary):
        return io.TextIOWrapper(binary, encoding="ascii", errors="replace")

    if magic.startswith(_MAGIC_COMPRESS):
        raise ValueError(
            f"{fn} is a unix-compress (.Z) file, which cannot be read; "
            "decompress it first (e.g. 'gzip -d' or 'uncompress')"
        )
    if magic.startswith(_MAGIC_GZIP):
        opener = gzip.open
    elif magic.startswith(_MAGIC_BZ2):
        opener = bz2.open
    elif magic.startswith(_MAGIC_XZ):
        opener = lzma.open
    elif zipfile.is_zipfile(fn):
        with zipfile.ZipFile(fn, "r") as z:
            names = [n for n in z.namelist() if ".sp3" in n.lower()]
            if not names:
                names = [n for n in z.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError(f"{fn} is a zip archive with no SP3 member")
            with z.open(names[0]) as member, text(member) as f:
                yield from f
        return
    else:
        opener = open

    with opener(fn, "rb") as fb, text(fb) as f:
        yield from f


def _sp3_epoch(line: str) -> pd.Timestamp:
    """Parse an SP3 epoch header line ``*  YYYY MM DD hh mm ss.ssssssss``."""
    parts = line[1:].split()
    if len(parts) < 6:
        raise ValueError(f"Malformed SP3 epoch line: {line!r}")
    year, month, day, hour, minute = (int(p) for p in parts[:5])
    return pd.Timestamp(year, month, day, hour, minute) + pd.Timedelta(
        seconds=float(parts[5])
    )


def _sp3_vector(line: str) -> tuple[float, float, float] | None:
    """The three F14.6 fields of a ``P`` or ``V`` record (columns 5-46).

    Returns None when a field is blank, short or unparsable, or holds a bad or
    absent value (999999.999999); for a ``P`` record a position of exactly
    0, 0, 0 (the SP3 "absent position" convention) is also bad.
    """
    try:
        vec = (float(line[4:18]), float(line[18:32]), float(line[32:46]))
    except ValueError:  # blank or short line, or garbage in a numeric field
        return None
    if any(abs(c) >= _SP3_BAD for c in vec):
        return None
    if line[0] == "P" and vec == (0.0, 0.0, 0.0):
        return None
    return vec


def df_sp3(fn: str | pathlib.Path, **kwargs) -> pd.DataFrame:
    """Read an SP3-c/d orbit file (plain text, gzip, bzip2, xz or zip).

    Every ``P`` (position) and ``V`` (velocity) record is attached to the epoch
    of the most recent ``*`` line, so files with many satellites per epoch are
    supported. Each position is matched with the velocity of the same
    satellite at the same epoch. Correlation records (``EP``/``EV``), header
    and comment lines and clock values are ignored; parsing stops at ``EOF``.
    The file is streamed line by line.

    Parameters
    ----------
    fn : str | pathlib.Path
        Path to the SP3 file. Compression is detected from the file contents.
        Unix-compress (``.Z``, LZW) files are not supported.
    **kwargs
        Not supported. Accepted only so that ``Spacecraft._load_table`` can
        forward ``read_kwargs``; any keyword raises ``TypeError``.

    Returns
    -------
    pd.DataFrame
        Columns ``sat`` (3-character SP3 vehicle ID, e.g. ``'L48'``), ``time``
        (epoch in the file's time system), ``x, y, z`` (m) and ``vx, vy, vz``
        (m/s); one row per (satellite, epoch).

    Raises
    ------
    TypeError
        If any keyword argument is passed.
    ValueError
        If the file has no velocity (``V``) records, a record appears before
        the first epoch line, or the file is unix-compressed.

    Notes
    -----
    SP3 positions are in km and velocities in dm/s; they are converted to m
    and m/s. A record that is bad or absent (blank, short or unparsable
    fields, a component of 999999.999999, or a position of exactly 0, 0, 0) is
    dropped with a logged warning naming its line numbers; a position left
    without a matching velocity is dropped too. A ``V`` record directly after
    the ``P`` record of the same satellite and epoch forms a pair with it, and
    a bad record drops its whole pair. If a (satellite, epoch) record appears
    more than once, the first good one is kept.
    """
    if kwargs:
        raise TypeError(
            "df_sp3 does not support keyword arguments; got "
            f"{sorted(kwargs)} (check Spacecraft.load_from_file read_kwargs)"
        )
    fn = pathlib.Path(fn)

    Key = tuple[str, pd.Timestamp]
    pos: dict[Key, tuple[float, float, float]] = {}
    vel: dict[Key, tuple[float, float, float]] = {}
    bad_lines: list[int] = []
    n_dup = 0
    seen_v = False
    epoch = None
    last_p: tuple[Key, bool] | None = None

    for lineno, line in enumerate(_sp3_lines(fn), start=1):
        if line.startswith("EOF"):
            break
        rec = line[:1]
        if rec == "*":
            epoch = _sp3_epoch(line)
        elif rec in ("P", "V"):
            if epoch is None:
                raise ValueError(f"{fn}: SP3 {rec} record before the first epoch")
            key = (line[1:4].strip(), epoch)
            vec = _sp3_vector(line)
            if rec == "P":
                # (key, accepted) of this P, for the V record that follows it
                last_p = (key, vec is not None and key not in pos)
                if vec is None:
                    bad_lines.append(lineno)
                elif key in pos:
                    n_dup += 1
                else:
                    pos[key] = vec
                continue
            seen_v = True
            paired = last_p is not None and last_p[0] == key
            if paired and not last_p[1]:
                continue  # its P record was bad or a repeat: drop the pair
            if vec is None:
                bad_lines.append(lineno)
                if paired:
                    del pos[key]  # the pair is bad; a later repeat may be good
                continue
            if key in vel:
                n_dup += 1
                continue
            vel[key] = vec

    if not seen_v:
        raise ValueError(
            f"{fn} has no velocity (V) records; Spacecraft requires velocities"
        )
    if bad_lines:
        logger.warning(
            "df_sp3: dropped %d bad/absent SP3 record(s) in %s at line(s) %s%s",
            len(bad_lines),
            fn,
            ", ".join(map(str, bad_lines[:10])),
            ", ..." if len(bad_lines) > 10 else "",
        )
    if n_dup:
        logger.warning(
            "df_sp3: ignored %d repeated (satellite, epoch) record(s) in %s; "
            "kept the first",
            n_dup,
            fn,
        )

    rows = []
    n_unmatched = 0
    for key, (x, y, z) in pos.items():
        v = vel.get(key)
        if v is None:
            n_unmatched += 1
            continue
        # km -> m, dm/s -> m/s
        rows.append(
            (*key, x * 1e3, y * 1e3, z * 1e3, v[0] / 10.0, v[1] / 10.0, v[2] / 10.0)
        )
    if n_unmatched:
        logger.warning(
            "df_sp3: dropped %d position record(s) without a matching velocity "
            "in %s",
            n_unmatched,
            fn,
        )

    return pd.DataFrame(rows, columns=["sat", "time", "x", "y", "z", "vx", "vy", "vz"])
