"""Derive SRP accelerations for an Earth orbiting spacecraft using
the General Mission Analysis Tool (GMAT).

added: 23/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
from pathlib import Path
import os


import numpy as np
import numpy.typing as npt
import pandas as pd

import contigo.config as config

from contigo.forces.base import ForceModel
from contigo.forces import srp_utils
from contigo.constellation import Constellation
from contigo.solar_system_ephem import SolarSystemEnvironment

class SRPGMATAcc:
    """Deriving *Cannonball* SRP acceleration from GMAT
    """

    def __init__(self,
                 sc_state: npt.ArrayLike | None = None,
                 sc_time: npt.ArrayLike | None = None,
                 sc_cr: npt.ArrayLike | None = None,
                 sc_srparea: npt.ArrayLike | None = None,
                 sc_mass:  npt.ArrayLike | None = None,
                 apistartup: str | None = None,
                 gmat_install: str | None = None):
        """Initialize the SRPGMATAcc class for deriving SRP acceleration from the
        General Mission Analysis Tool (GMAT).

        This instantiation uses the Cannonball method from the GMAT API.

        Both the ECEF and ECI accelerations are derived in km/s^2 internally (GMAT units).

        Parameters
        ----------
        sc_state : npt.ArrayLike | None, optional
            Spacecraft state vector (N,[x,y,z,vx,vy,vz]), by default None.
        sc_time : npt.ArrayLike | None, optional
            Spacecraft time (N,), by default None.
        sc_cr : npt.ArrayLike | None, optional
            Spacecraft coeffecient of reflection (N,), by default None.
            
        sc_srparea : npt.ArrayLike | None, optional
           The area used to compute acceleration due to solar radiation pressure 
           for the spherical SRP area model, m^2 (N, ). By default None
        sc_mass : npt.ArrayLike | None, optional
           Spacecraft mass, kg (N, ), by default None
        apistartup : str | None, optional
            GMAT startup file for loading and adding GMAT to the python path.
        gmat_install : str | None, optional
            GMAT installation  directory for adding GMAT to the python path.
        """
        #first need to setup and load GMAT Python API
        srp_utils.setup_gmat(apistartup,gmat_install)

        self.state = sc_state
        self.time = sc_time
        self.cr = sc_cr
        self.srp_area = sc_srparea
        self.mass = sc_mass

        self.srp_acc_ecef = None
        self.srp_acc_eci = None

    def calc_srp(self):
        """Calculate ECEF and ECI SRP acceleration
        """
        gmat = config.state['gmatpy']

        gtime = pd.to_datetime(np.array(self.time)).strftime('%d %b %Y %H:%M:%S.000')
        #setup a gmat spacecraft
        earthorb = gmat.Construct("Spacecraft", "EarthOrbiter")
        earthorb.SetField("DateFormat", "TAIGregorian")
        earthorb.SetField("CoordinateSystem", "EarthFixed")

        # Create the converter
        csConverter = gmat.CoordinateConverter()

        # Create the input and output coordinate systems
        eci  = gmat.Construct("CoordinateSystem", "ECI", "Earth", "MJ2000Eq")
        ecef = gmat.Construct("CoordinateSystem", "ECEF", "Earth", "BodyFixed")
        

        # Solar Radiation Pressure
        srp = gmat.Construct("SolarRadiationPressure", "SRP")

        # Add all of the forces into the ODEModel container
        # ODE Model settings
        fm = gmat.Construct("ForceModel", "FM")
        fm.SetField("CentralBody", "Earth")
        fm.AddForce(srp)

        # Setup the state vector used for the force
        psm = gmat.PropagationStateManager()
        psm.SetObject(earthorb)
        psm.BuildState()

        fm.SetPropStateManager(psm)
        fm.SetState(psm.GetState())

        gmat.Initialize()

        srp_der = []
        srp_der_ecef = []

        for st, ep, cr, srp_a, mass in zip(self.state,gtime,self.cr, self.srp_area, self.mass):
            earthorb.SetField("Epoch", f"{ep}")
            earthorb.SetField("X", st[0])
            earthorb.SetField("Y", st[1])
            earthorb.SetField("Z", st[2])
            earthorb.SetField("VX", st[3])
            earthorb.SetField("VY", st[4])
            earthorb.SetField("VZ", st[5])
            earthorb.SetField("DryMass", mass)
            earthorb.SetField("Cr", cr)
            earthorb.SetField("SRPArea", srp_a)

            gmat.Initialize()

            # Finish force model setup:
            ##  Map spacecraft state into the model
            fm.BuildModelFromMap()
            ##  Load physical parameters needed for the forces
            fm.UpdateInitialData()

            pstate = earthorb.GetState().GetState()

            srp.GetDerivatives(pstate)
            srp_der.append(srp.GetDerivativeArray())

            # rotate the accelerations into ecef
            conv_vec = gmat.Rvector6(0,0,0,0,0,0)

            # SRP
            csConverter.Convert(earthorb.GetEpoch(), 
                                gmat.Rvector6(srp_der[-1]), eci, 
                                conv_vec, ecef)
            srp_der_ecef.append(conv_vec.GetDataVector())

        srp_der_ecef = np.array(srp_der_ecef)[:,-3:]
        srp_der = np.array(srp_der)[:,-3:]

        self.srp_acc_ecef = srp_der_ecef
        self.srp_acc_eci = srp_der

    def get_ecef_acc(self):
        """Return ECEF acceleration
        """        
        return self.srp_acc_ecef
    
    def get_eci_acc(self):
        """Return ECI acceleration
        """  
        return self.srp_acc_eci

    def get_all_acc(self):
        """Return ECEF and ECI acceleration
        """  
        return self.srp_acc_ecef, self.srp_acc_eci


class SRPAcc(ForceModel):
    """
    SRP accelerations for individual satellites in a Constellation object.
    """

    name: str = "SRPGMATCannonball"

    def __init__(self, 
                 apistartup: str | None = None, 
                 gmat_install: str | None = None):
        """SRP acceleration for individual satellets in a Constellation object.

        Wrapper for SRPGMATAcc which follows the .base.ForceModel(Protocol)

        Parameters
        ----------
        apistartup : str | None, optional
            GMAT startup file for loading and adding GMAT to the python path.
        gmat_install : str | None, optional
            GMAT installation  directory for adding GMAT to the python path.
        """        
        self.apistartup = apistartup
        self.gmat_install = gmat_install

    def acceleration(self, 
                     constellation: Constellation,
                     solarsys_env: SolarSystemEnvironment
                     ) -> dict[str, npt.NDArray[np.float64]]:
        """Derive SRP accelerations. 

        Use SRPGMATAcc to derive *cannonball* SRP accelerations for spacecraft in 
        a Constellation object.

        Constellation holds the state and time for all satellites. 

        Parameters
        ----------
        constellation : Constellation
            Constellation container of Spacecraft objects.

        Returns
        -------
        dict[spacecraft_id] -> (N,3)
        """        
        acc_dict = {}
        for sc_id, sc in constellation.spacecraft.items():

            srp = SRPGMATAcc(sc_state=sc.state_ecef / 1000.0,  # m → km for GMAT
                             sc_time=sc.time,
                             sc_cr=sc.cr_arr,
                             sc_srparea=sc.srp_area_arr,
                             sc_mass=sc.sc_mass_arr,
                             apistartup=self.apistartup,
                             gmat_install=self.gmat_install)
            srp.calc_srp()
            acc_dict[sc_id] = srp.get_ecef_acc() * 1000.0  # km/s² → m/s²

        return acc_dict

    def potential(self, 
                constellation: Constellation
                ) -> dict[str, npt.NDArray[np.float64]]:
        """Not implemented for SRPAcc, only acceleration is calculated.
        """        
        raise NotImplementedError("Not implemented for SRPAcc.")