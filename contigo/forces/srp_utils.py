"""Setup GMAT API.

added: 23/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging
import os
import sys
from pathlib import Path

import contigo.config as config

logger = logging.getLogger(__name__)


def setup_gmat(apistartup: str | Path, gmat_install: str | Path):
    """Setup the GMAT python API.

    Parameters
    ----------
    apistartup : str | Path
        GMAT startup file for loading and adding GMAT to the python path.
    gmat_install : str | Path
        GMAT installation  directory for adding GMAT to the python path.

    Raises
    ------
    ValueError
        Cannot find the GMAT startup file in the GMAT installation directory.
    """
    gmat_bin_path = os.path.join(gmat_install, "bin")
    startup = os.path.join(gmat_bin_path, apistartup)

    if os.path.exists(startup) and config.state["gmat_loaded"] is False:
        logger.info("Setting up GMAT API.")
        sys.path.insert(1, gmat_bin_path)
        import gmatpy as gmat

        gmat.Setup(startup)

        config.state["gmat_loaded"] = True
        config.state["gmatpy"] = gmat

    elif config.state["gmat_loaded"]:
        logger.info("GMAT API already setup.")
        import gmatpy as gmat

        config.state["gmatpy"] = gmat
    else:
        raise ValueError(f"Cannot find {startup} in {gmat_bin_path}")
