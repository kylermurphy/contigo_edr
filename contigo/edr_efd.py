"""Class for deriving Energy Dissipation Rate and Effective Density.

added: 30/03/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import pandas as pd
import numpy as np

from scipy.integrate import cumulative_simpson

import contigo.contigo_utils.constants as constants

from contigo.forces.base import ForceModel
from contigo.constellation import Constellation
from contigo.solar_system_ephem import SolarSystemEnvironment


class EDRDensity:
    """
    Energy Dissipation Rate and Effiective Density calculator.

    Accepts:
        - Constellation
        - SolarSystemEnvironment
        - list of ForceModel implementations

    Computes:
        - Effective density
        - Energy dissipation rate 
        - Denominator of the effective density calculation
        - Solar System Ephemeris
        - Accelerations
        - Graviational potential
    """

    def __init__(self,
                 constellation: Constellation,
                 solarsys_env: SolarSystemEnvironment,
                 force_models: list[ForceModel],
                 potential_model: ForceModel,):
        """Intialize the EDRDensity calculator. 

        Parameters
        ----------
        constellation : Constellation
            Constellation container with the spcaecraft state, physical properties,
            spacecraft IDs, and time arrays.
        solarsys_env : SolarSystemEnvironment
            Solar System Environment container to load the ephemeris of solar system 
            bodies. The ephemeris is loaded for the derivation of the third body 
            accelerations.
        force_models : list[ForceModel]
            List of ForceModel implementations to compute accelerations
        potential_model : ForceModel
            Gravity potential model to use in the derivation of the gravity potential
            used in the EDR calculation.
        """
        self.constellation = constellation
        self.solarsys_env = solarsys_env
        self.force_models = force_models
        self.potential_model = potential_model

        unique_gps, u_id = np.unique(constellation.sspice_gps, return_index=True)

        # load ephemeris for all unique times
        self.solarsys_env._load_times(ephem_time=constellation.sspice_et[u_id], 
                                      gps_time=unique_gps, 
                                      utc_time=constellation.sc_utc)


    def compute_den(self,
                    window: pd.Timedelta=pd.Timedelta(minutes=90),
                    smth_edr: int=10,
                    smth_den: int | None = None,) -> dict:
        """Derive the effective density, equation 5 and A16 of 
        https://doi.org/10.1029/2024EA003898, from the EDR and the denominator.

        Parameters
        ----------
        window : pd.Timedelta, optional
            The size of the window used for deriving the deltas of the EDR and 
            denominator, by default pd.Timedelta(minutes=90)
        smth_edr : int, optional
            The number of points to smooth edr returned by compute_edr(), by default 10
        smth_den : int | None, optional
            The number of points to smooth the final effective density, by default None

        Returns
        -------
        dict
            A dictionary keyed by spacecraft ID with the effective density time series.
            The timeseries is a pandas DataFrame with columns 'DateTime' and 'efd' for
            the effective density. The 'DateTime' column is the center of the time
            window used for deriving the effective density.

        Raises
        ------
        ValueError
            The keys (identifying the spacecraft) of the EDR and denominator must match.
        ValueError
            The shape of the EDR and denominator time series of each spacecraft must
            match.
        """

        # create dictionary for effective density results 
        self.efd = {}
        # compute edr and denomentator from A18 and A19 of 
        # https://doi.org/10.1029/2024EA003898
        self.edr = self.compute_edr()
        self.denom = self.compute_denom()

        # derive the effective density from edr (numerator) and the denominator
        # as in equation 5 and and A16 of https://doi.org/10.1029/2024EA003898

        edr_k = self.edr.keys()
        den_k = self.denom.keys()

        if set(edr_k) != set(den_k):
            raise ValueError('EDR and denominator keys do not match')
        
        for sc_id in edr_k:
            edr = self.edr[sc_id]
            denom = self.denom[sc_id]

            if edr.shape != denom.shape:
                raise ValueError(f'EDR and denominator shapes do ' \
                'not match for sc_id {sc_id}')
            
            # smooth the edr
            edr_sm = pd.Series(edr).rolling(smth_edr, min_periods=1, center=True).mean().to_numpy() 

            n_win = self.constellation[sc_id].sc_utc[-1] - \
                    self.constellation[sc_id].sc_utc[0]
            n_win = int(n_win.total_seconds()/window.total_seconds())+10

            id0 = np.zeros(n_win, dtype=int)
            id1 = np.zeros(n_win, dtype=int)

            # start and end time for the effective density calculation
            s = self.constellation[sc_id].sc_utc[0]
            e = s+window

            # get the indices of the window start and end times for subtractions
            i = 0
            while e < self.constellation[sc_id].sc_utc[-1]:
                # find the indices of the edr time series that fall within the window
                idx = np.where((self.constellation[sc_id].sc_utc >= s) & 
                               (self.constellation[sc_id].sc_utc < e))

                id0[i] = idx[0].min()
                id1[i] = idx[0].max()
                i +=1

                s = s+window
                e = e+window

            delta_edr = edr_sm[id1[0:i]] - edr_sm[id0[0:i]]
            delta_den = denom[id1[0:i]] - denom[id0[0:i]]

            efd_sat = -2*delta_edr/delta_den
            efd_t = self.constellation[sc_id].sc_utc[id1[0:i]] + \
                    (self.constellation[sc_id].sc_utc[id1[0:i]] - \
                    self.constellation[sc_id].sc_utc[id1[0:i]])/2.

            if smth_den is not None:
                efd_sat = pd.Series(efd_sat).rolling(smth_den, min_periods=1, 
                                                     center=True).mean().to_numpy()
            
            efd_df = pd.DataFrame({'DateTime':efd_t, 'efd':efd_sat})

            self.efd[sc_id] = efd_df


        return self.efd
        

    def compute_edr(self) -> dict:
        """
        Compute EDR (Delta Epsilon) from equation A18 and A19 of
        https://doi.org/10.1029/2024EA003898

        Returns
        -------
        dict
            Dictionary mapping spacecraft IDs to EDR time series.

        Notes
        -----
        Combines force model accelerations and gravitational potential
        to compute energy dissipation.
        """
        # system is the constellation of spacecraft
        spacecraft_dict = self.constellation.spacecraft

        results = {}

        earth_angv2 = constants.WGS84_EARTH_ANGULAR_VELOCITY**2

        # derive the earth potential for each sc in the 
        # constellation
        e_gp_con = self.potential_model.potential(self.constellation)

        self.e_gp = e_gp_con

        # derive the accelerations from the force models
        # for the constellation
        acc_con = { }
        for model in self.force_models:
            acc_con[model.name] = model.acceleration(self.constellation, self.solarsys_env)

        self.accelerations = acc_con

        for sc_id, sc in spacecraft_dict.items():
            N = sc.N
            
            sc_p = sc.state_ecef[:, 0:3]
            sc_v = sc.state_ecef[:, 3:]

            sc_xy2 = sc_p[:,0]**2 + sc_p[:,1]**2
            sc_v2 = (sc_v*sc_v).sum(axis=1)

            e_gp = e_gp_con[sc_id]

            # equation a18 and a19 non integral portion
            # here we use r^2*cos*2(phi) = x^2+y^2
            # and we subtract the edr at edr time zero
            edr = sc_v2/2. - e_gp - earth_angv2*sc_xy2/2.

            # compute the acceleration integrals
            # x-axis for integrating
            acc_int = np.zeros(N)
            x_ax = sc.sspice_gps
            x_ax = x_ax-x_ax.min()
            for m_id, m_acc in acc_con.items():
                # if the force model returns multiple accelerations
                # loop through them all
                if m_acc[sc_id].shape[0] != N:
                    for i in range(m_acc[sc_id].shape[0]):
                        acc = m_acc[sc_id][i,:,:]
                        if acc.shape[0] != N:
                            raise ValueError('Model accelerations should be shape (N,3) or (x,N,3)')
                        a_int = (acc*sc_v).sum(axis=1)
                        acc_int += cumulative_simpson(a_int, x=x_ax, initial=0)
                else:
                    a_int = (m_acc[sc_id]*sc_v).sum(axis=1)
                    acc_int += cumulative_simpson(a_int, x=x_ax, initial=0)

            edr = edr - acc_int
            edr = edr - edr[0] 

            results[sc_id] = edr

        return results
    
    def compute_denom(self) -> dict:
        """
        Compute denomenator of equation A20 of
        https://doi.org/10.1029/2024EA003898

        Returns
        -------
        Dictionary keyed by spacecraft ID.
        """
        # system is the constellation of spacecraft
        spacecraft_dict = self.constellation.spacecraft

        denom = {}

        for sc_id, sc in spacecraft_dict.items():
            
            b=sc.cd_arr*sc.drag_area_arr/sc.sc_mass_arr
            
            sc_v = sc.state_ecef[:, 3:]
            sc_v3 = np.linalg.norm(sc_v,axis=1)**3

            x_ax = sc.sspice_gps
            x_ax = x_ax-x_ax.min()

            y_int = b*sc_v3

            denom[sc_id] = cumulative_simpson(y_int,x=x_ax,initial=0)

        return denom