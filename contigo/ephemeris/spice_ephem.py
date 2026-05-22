"""Ephemeris provider using SPICE and SPICEYPY.

added: 04/03/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import posixpath
import urllib.parse
import logging

from os import path
from operator import itemgetter
from datetime import datetime, timezone
from dateutil import tz


import numpy as np
import numpy.typing as npt

import spiceypy as spice

import contigo.contigo_utils.utils as utils
import contigo.config as config

from contigo.ephemeris.base import EphemerisProvider
from contigo.contigo_utils.constants import GMc

logger = logging.getLogger(__name__)

class SPICEEphem(EphemerisProvider):
    """
    SPICE-backed ephemeris provider with unique-time optimization.
    """

    def __init__(self, 
                 ephemeris: str='de440s',
                 frame: str = "ITRF93",
                 observer: str = "EARTH"):

        self.frame = frame
        self.observer = observer
        self.ephemeris= ephemeris

        self.load_kernels()

    def __call__(self, body: npt.NDArray[np.str_] | list[str],
                 utc_time:  None = None,
                 gps_time: None = None,
                 ephem_time: np.ndarray | None = None,
                 ):
        """
        Compute positions of celestial bodies.

        Parameters
        ----------
        ephem_time : npt.NDArray[np.float64]
            Ephemeris time array (ET) in seconds.
        gps_time : npt.NDArray[np.float64]
            GPS time array corresponding to ephem_time.
        utc_time : npt.NDArray[np.datetime64]
            UTC datetime array corresponding to ephem_time.

            
        Returns
        -------
        np.ndarray
            Array of shape (n_bodies, N, 3) containing position vectors.

        Notes
        -----
        This provider only uses ephem_time for its internal logic, but the interface 
        allows for flexibility in case future providers need to use different
        time systems.
        """
        # find the unique values of et and return their indecies
        # return the inverse indices of that allow the reconstruction
        # of the orignal array
        unique_et, inv = np.unique(ephem_time, return_inverse=True)

        r_unique = np.array(
            [
            spice.spkpos(bd.upper(), unique_et, self.frame,'NONE',self.observer)[0]
            for bd in body
            ]) * 1000.0  # SPICE returns km; convert to m

        r_body = r_unique[:,inv,:]
        return r_body
    
    def load_kernels(self):
        """Download and load the SPICE kernels for deriving body locations

        Checks if an attemp at downloading the kernels has already been done.

        Download will check if files have changed and need to be redownloaded.
        Checks if kernels are already loaded.

        Loads the kernels.

        Currently loads leap seconds (.tls), ephemeris (.bsp), and Earth orientation 
        data (.bpc)
        """
        # set file names 
        ephem_f = f'{self.ephemeris}.bsp'
        leaps_f = config.LEAP_FILE
        pck_f = config.PCK_FILE
        # get file paths
        sp_kernels = [path.join(config.DATA_DIR,fp) for fp in [ephem_f,leaps_f,pck_f]]

        # check if we've already attempted to download kernels
        if config.state['kernel_downloaded'] is False:
            self.dl_kernels(ephem_f, leaps_f, pck_f)

        # check if kernels are already loaded, load them if not
        sp_kcnt = spice.ktotal('ALL')
        sp_loaded = [spice.kdata(i,'ALL')[0] for i in range(sp_kcnt)]
        for fp in sp_kernels:
            if fp in sp_loaded:
                logger.info('Kernel already loaded - %s', fp)
            else:
                logger.info('Loading Kernel - %s', fp)
                spice.furnsh(fp) # need to check if kernels are loaded

    def dl_kernels(self, ephem_f: str, leaps_f: str, pck_f:str):
        """Download required JPL SPICE kernels.

        Download ephemeris, leap second, and Earth orientation Kernels from JPLs 
        Navigation and Ancillary Information Facility (NAIF). 

        Will check if local files exist. Will download if files on the server are newer
        then the local files.

        Parameters
        ----------
        ephem_f :str
            JPL ephemeris file to download.
        leaps_f : str
            JPL leap seconds file to download.
        pck_f : str
            JPL Earth orientation file to download.
        """
        base_url = 'https://naif.jpl.nasa.gov'
        base_pth = '/pub/naif/generic_kernels'
        
        ephem_d = 'spk/planets'
        leaps_d = 'lsk'
        pck_d = 'pck'

        ephem_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,ephem_d,ephem_f))
        leaps_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,leaps_d,leaps_f))
        pck_url = urllib.parse.urljoin(base_url,
                                    posixpath.join(base_pth,pck_d,pck_f))

        for url, file in zip([ephem_url,leaps_url,pck_url],[ephem_f,leaps_f,pck_f]):
            # create file names
            fp = path.join(config.DATA_DIR,file)
            if not path.exists(fp):
                logger.info('Downloading kernel - %s', fp)
                utils.dl_file(url,fp)
            else:
                #check for modification times
                loc_tz = datetime.now().astimezone().tzinfo
                gmt_tz = tz.gettz('GMT')

                mod_file = datetime.fromtimestamp(path.getmtime(fp), tz=loc_tz)
                mod_file = mod_file.astimezone(gmt_tz)
                mod_url = utils.wf_mtime(url)

                if mod_url == None:
                    logger.info('Could not determine modification time of %s', url)
                elif mod_url > mod_file:
                    logger.info('Downloading new version of %s', url)
                    utils.dl_file(url,fp)
        
        config.state['kernel_downloaded'] = True