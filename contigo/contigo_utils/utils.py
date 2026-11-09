"""Utilites for the contigo module

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import functools
import gzip
import io
import logging
import pathlib
import shutil
import zipfile
from datetime import datetime

import pandas as pd
import requests
from dateutil import tz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dl_file(url, filename):
    """
    Download file from url to filename

    Parameters
    ----------
    url : url to file.
    filename : filename to save as.

    Raises
    ------
    for
        DESCRIPTION.
    RuntimeError
        DESCRIPTION.

    """
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    # if requests couldn't get file size drop encoding to get it
    if file_size == 0:
        rs = requests.head(url, headers={"Accept-Encoding": None})
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


def wf_mtime(url):
    """
    Retrieves the last modified date of a file on the web.

    Args:
        url (str): The URL of the file.

    Returns:
        datetime or None: The last modified date as a datetime object,
                          or None if the header is not found or an error occurs.
    """
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
        print(f"Error fetching URL: {e}")
        return None


#: SP3 "bad or absent value" sentinel (clock and, in practice, other fields)
_SP3_BAD = 999999.0


def _sp3_lines(fn: pathlib.Path) -> list[str]:
    """Return the text lines of a plain, gzip or zip SP3 file.

    The compression is detected from the file's magic bytes, not its name.
    Unix ``compress`` (``.Z``) files cannot be read by :mod:`gzip`, so they
    raise a clear error instead of being decoded as garbage.
    """
    with open(fn, "rb") as fb:
        magic = fb.read(2)

    if magic == b"\x1f\x8b":  # gzip (.gz, and .Z files that are really gzip)
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            return f.read().splitlines()
    if magic == b"\x1f\x9d":  # unix compress (LZW)
        raise ValueError(
            f"{fn} is a unix-compress (.Z) file, which cannot be read; "
            "decompress it first (e.g. 'gzip -d' or 'uncompress')"
        )
    if zipfile.is_zipfile(fn):
        with zipfile.ZipFile(fn, "r") as z:
            names = [n for n in z.namelist() if ".sp3" in n.lower()]
            if not names:
                names = [n for n in z.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError(f"{fn} is a zip archive with no SP3 member")
            with z.open(names[0]) as member:
                return io.TextIOWrapper(member, encoding="utf-8").read().splitlines()
    with open(fn, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _sp3_epoch(line: str) -> pd.Timestamp:
    """Parse an SP3 epoch header line ``*  YYYY MM DD hh mm ss.ssssssss``."""
    parts = line[1:].split()
    if len(parts) < 6:
        raise ValueError(f"Malformed SP3 epoch line: {line!r}")
    year, month, day, hour, minute = (int(p) for p in parts[:5])
    return pd.Timestamp(year, month, day, hour, minute) + pd.Timedelta(
        seconds=float(parts[5])
    )


def _sp3_vector(line: str) -> tuple[float, float, float]:
    """The three F14.6 fields of a ``P`` or ``V`` record (columns 5-46)."""
    return (float(line[4:18]), float(line[18:32]), float(line[32:46]))


def df_sp3(fn: str | pathlib.Path, **kwargs) -> pd.DataFrame:
    """Read an SP3-c/d orbit file (plain text, gzip or zip).

    Every ``P`` (position) and ``V`` (velocity) record is attached to the epoch
    of the most recent ``*`` line, so files with many satellites per epoch are
    supported. Each position is matched with the velocity of the same
    satellite at the same epoch. Correlation records (``EP``/``EV``), header
    and comment lines and clock values are ignored; parsing stops at ``EOF``.

    Parameters
    ----------
    fn : str | pathlib.Path
        Path to the SP3 file. Compression (gzip or zip) is detected from the
        file contents. Unix-compress (``.Z``) files are not supported.
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
    and m/s. Records with a bad or absent position or velocity (a component
    of 999999.999999, or a position of exactly 0, 0, 0) are dropped with a
    logged warning, as are positions without a matching velocity.
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
    bad: set[Key] = set()
    epoch = None

    for line in _sp3_lines(fn):
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
            if any(abs(c) >= _SP3_BAD for c in vec) or (
                rec == "P" and vec == (0.0, 0.0, 0.0)
            ):
                bad.add(key)
            (pos if rec == "P" else vel)[key] = vec

    if not vel:
        raise ValueError(
            f"{fn} has no velocity (V) records; Spacecraft requires velocities"
        )

    rows = []
    n_unmatched = 0
    for key, p in pos.items():
        if key in bad:
            continue
        if key not in vel:
            n_unmatched += 1
            continue
        rows.append((key[0], key[1], *p, *vel[key]))
    if bad:
        logger.warning(
            "df_sp3: dropped %d SP3 record(s) with bad/absent values in %s",
            len(bad),
            fn,
        )
    if n_unmatched:
        logger.warning(
            "df_sp3: dropped %d position record(s) without a matching velocity "
            "in %s",
            n_unmatched,
            fn,
        )

    df = pd.DataFrame(rows, columns=["sat", "time", "x", "y", "z", "vx", "vy", "vz"])
    df[["x", "y", "z", "vx", "vy", "vz"]] = df[
        ["x", "y", "z", "vx", "vy", "vz"]
    ].astype(float)
    df[["x", "y", "z"]] *= 1000.0  # km -> m
    df[["vx", "vy", "vz"]] /= 10.0  # dm/s -> m/s
    df["time"] = pd.to_datetime(df["time"])
    return df
