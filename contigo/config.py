"""Config and State values for CONTIGO

added: 19/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging

from os import makedirs
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def data_path():
    # setup data dir 
    # find directory of module
    # module directory/swdata/ is where the data is stored
    f_path = Path(__file__).resolve()
    d_dir = f_path / '..' / 'data'
    d_dir = d_dir.resolve()

    # create it if it doesn't exist
    if not d_dir.exists():
        makedirs(d_dir)

    return str(d_dir)

# Static configuration values (constants)
DATA_DIR = data_path()
LEAP_FILE = 'naif0012.tls'
PCK_FILE = 'earth_latest_high_prec.bpc'

# Mutable State Values
state = {'orekit_loaded':False,
         'gmat_loaded':False,
         'gmatpy':None,
         'kernel_downloaded':False,
         'pot_coef_loaded':False,
         'pot_file': None, 
         'pot_clm': None,
         'pot_slm': None,
         'pot_r0': None,
         'pot_GM': None,}
