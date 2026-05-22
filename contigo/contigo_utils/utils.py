"""Utilites for the contigo module

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import pathlib
import functools
import shutil
import requests
import zipfile
import gzip
import io

from datetime import datetime
from dateutil import tz
from tqdm import tqdm

import pandas as pd

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
    file_size = int(r.headers.get('Content-Length', 0))
    
    # if requests couldn't get file size drop encoding to get it
    if file_size == 0:
        rs = requests.head(url, headers={'Accept-Encoding': None})
        file_size = int(rs.headers.get('Content-Length', 0))
        
    
    fpath = pathlib.Path(filename).expanduser().resolve()
    fpath.parent.mkdir(parents=True, exist_ok=True)
    
    desc = "(Unknown total file size)" if file_size == 0 else ""
    post = f'Downloading: {url} to {filename}'
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc, postfix=post, position=0, leave=True) as r_raw:
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

        last_modified_header = response.headers.get('Last-Modified')

        if last_modified_header:
            # Parse the date string from the header
            # Example format: 'Wed, 21 Oct 2015 07:28:00 GMT'
            return datetime.strptime(
                last_modified_header, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=tz.gettz('GMT'))
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    
def df_sp3(fn):
    """
    Read SP3c files (Standard, Gzip, or Zip).
    """    
    # 1. Handle Gzip (.gz or .Z)
    if fn.lower().endswith(('.gz', '.z')):
        f = io.TextIOWrapper(gzip.open(fn, 'rb'), encoding='utf-8')
        
    # 2. Handle Zip (.zip)
    elif zipfile.is_zipfile(fn):
        with zipfile.ZipFile(fn, 'r') as z:
            internal_fn = [n for n in z.namelist() if '.sp3' in n.lower()][0]
            f = io.TextIOWrapper(z.open(internal_fn), encoding='utf-8')
            
    # 3. Handle Plain Text
    else:
        f = open(fn, 'r', encoding='utf-8')

    dt, dat, vel = [], [], []

    try:
        # The parsing logic remains the same
        for line in f:
            if line.startswith('*'): 
                dt.append(line[1:].strip())
            elif line.startswith('P'): 
                dat.append([line[0:4],line[4:18], line[18:32], line[32:46]])
            elif line.startswith('V'):
                vel.append([line[4:18], line[18:32], line[32:46]])
    finally:
        f.close()

    # Data Processing
    # (Note: Standardize the datetime format to match SP3 spacing)
    # If dt looks like "2023 10 27  0  0  0.00000000"
    dt = pd.to_datetime(dt, format='%Y %m %d %H %M %S.%f', errors='coerce')

    pdf = pd.DataFrame(data=dat, columns=['sat', 'x', 'y', 'z'])
    # SP3c position is in km; convert to m
    pdf[['x', 'y', 'z']] = pdf[['x', 'y', 'z']].astype(float) * 1000.0

    # Basic check: if you have 1 epoch but 32 satellites,
    # you'll need to repeat the 'dt' values to match 'dat' length.
    if len(dt) != len(pdf):
        # This is a placeholder; real SP3 logic usually requires
        # tracking the current epoch inside the loop.
        pass
    else:
        pdf['time'] = dt

    vdf = pd.DataFrame(data=vel, columns=['vx', 'vy', 'vz'])
    # SP3c velocity is in units of 10^-4 dm/s; convert to m/s
    vdf[['vx', 'vy', 'vz']] = vdf[['vx', 'vy', 'vz']].astype(float) / 10.0

    return pd.concat([pdf, vdf], axis=1)
