from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec, WGS72
from astropy.time import Time
import numpy as np

import parameters as params
from attitude import get_slew_data

class Satellite:

    def __init__(
        self,
        telescope_boresight=params.TELESCOPE_BORESIGHT,
        antennas=params.ANTENNAS,
        solar_array=params.SOLAR_ARRAY,
        sc_inertia=params.SC_INERTIA,
        max_wheel_momentum=params.MAX_WHEEL_MOMENTUM,
        max_wheel_torque=params.MAX_WHEEL_TORQUE,
        wheel_design_margin=params.WHEEL_DESIGN_MARGIN   
    ):
        """
        Create a Satellite instance.
        
        Parameters
        ----------
        telescope_boresight : numpy.ndarray
            A unit vector representing the telescope boresight direction 
            in spacecraft body-frame.

        antennas : list of numpy.ndarray
            A list of unit vectors each representing the antenna direction 
            in spacecraft body-frame.

        solar_array : numpy.ndarray
            A unit vector representing the average direction of all antenna normals 
            in spacecraft body-frame.

        sc_inertia : numpy.ndarray
            A 3x3 symmetric matrix representing the spacecraft's inertia tensor 
            (units: kg·m²), describing the mass distribution along each axis.
        
        max_wheel_momentum : float
            The maximum allowable angular momentum of the reaction wheel 
            (units: N·m·s).
        
        max_wheel_torque : float
            The maximum allowable torque of the reaction wheel (units: N·m).
        
        wheel_design_margin : float
            The design margin factor applied to the reaction wheel's performance limits 
            as a buffer to ensure safe operation.

        Attributes
        ----------
        telescope_boresight : numpy.ndarray
            A unit vector representing the telescope boresight direction 
            in spacecraft body-frame.

        antennas : list of numpy.ndarray
            A list of unit vectors each representing the antenna direction 
            in spacecraft body-frame.

        solar_array : numpy.ndarray
            A unit vector representing the average direction of all antenna normals 
            in spacecraft body-frame.

        sc_inertia : numpy.ndarray
            A 3x3 symmetric matrix representing the spacecraft's inertia tensor 
            (units: kg·m²), describing the mass distribution along each axis.
        
        max_wheel_momentum : float
            The maximum allowable angular momentum of the reaction wheel 
            (units: N·m·s).
        
        max_wheel_torque : float
            The maximum allowable torque of the reaction wheel (units: N·m).
        
        wheel_design_margin : float
            The design margin factor applied to the reaction wheel's performance limits 
            as a buffer to ensure safe operation.

        earth_satellite : skyfield.sgp4lib.EarthSatellite
             A Skyfield EarthSatellite object used to model the satellite's orbital
             trajectory and predict its position over time.

        slew_angle_data : list of float
            A list containing the slew angles (in degrees) for which the minimum 
            durations were calculated based on reaction wheel limitations.
    
        slew_time_data : list of float
            A list containing the minimum slew durations (in seconds) corresponding 
            to each slew angle in `slew_angle_data`, calculated based on reaction 
            wheel limitations.
        """
        # Spacecarft parameter attributes
        self.telescope_boresight = telescope_boresight
        self.antennas = antennas
        self.solar_array = solar_array
        self.sc_inertia = sc_inertia
        self.max_wheel_momentum = max_wheel_momentum
        self.max_wheel_torque = max_wheel_torque
        self.wheel_design_margin = wheel_design_margin
        
        # Initialize attributes
        self.earth_satellite = None
        self.slew_angle_data = None
        self.slew_time_data = None

        # Calculate the minimum slew durations required for slews rotating about Z-axis
        # with angles ranging from 0° to 180°, given the reaction wheel's limitations.
        self.update_slew_data()

    
    def update_slew_data(self):
        self.slew_angle_data, self.slew_time_data = get_slew_data(
            self.sc_inertia, self.max_wheel_momentum, 
            self.max_wheel_torque, self.wheel_design_margin
        )

    
    def __repr__(self):
        return (
            f"<Satellite object: max_wheel_momentum={self.max_wheel_momentum} N·m·s, "
            f"max_wheel_torque={self.max_wheel_torque} N·m>"
        )
        
    
    def build_from_params(
        self, 
        T0=params.T0, 
        ecco=params.ECCO, 
        argpo=params.ARGPO, 
        inclo=params.INCLO, 
        RAAN=params.RAAN, 
        period=params.PERIOD, 
        mo=params.MO
    ):
        """
        Create a Skyfield EarthSatellite object from orbital parameters.
        
        Orbit set-up: The reference plane is the Earth's equitorial plane. 
        The intersection between the reference plane and the orbital plane is the 
        line-of-nodes, connecting the ECI frame origin with the acending and 
        descending nodes.

        Note: The sgp4init() method from the SGP4 library uses days since 
        1949 December 31 00:00 UT as the epoch for satellite construction, which yields 
        a "dubious year" warning from ERFA (Essential Routines for Fundamental Astronomy). 
        Consider using an alternative constructor for better propagation accuracy.
    
        Parameters
        ----------
        T0 : astropy.time.Time
            Epoch time specifying the moment at which the orbital elements are defined. 
       
        ecco : float
            Eccentricity of the orbit. 
        
        argpo : float
            Argument of perigee is the angle in degrees measured from the ascending node 
            to the perigee; defines the orientation of the ellipse in the orbital plane.
        
        inclo : float
            Inclination is the vertical tilt in degrees of the orbital plane with respect 
            to the reference plane, measured at the ascending node.
        
        RAAN : float
            The right ascension of ascending node is the angle in degrees along the
            reference plane of the ascending node with respect to the reference frame’s
            vernal equinox. (i.e. horizontally orients the ascending node)
        
        period : float
            Orbital period in minutes.
        
        mo : float
            Mean anomaly is the angle in degrees from perigee defining the position of
            the satellite along the elipse at epoch T0.
            
        Returns
        -------
        skyfield.sgp4lib.EarthSatellite
             A Skyfield EarthSatellite object used to model the satellite's orbital
             trajectory and predict its position over time.
        """
        # Initiate the satellite object
        satrec = Satrec()
    
        # Define the `epoch` in sgp4 satrec 
        tau0 = Time("1949-12-31 0:0:0").jd 
        tau = T0.jd
        epoch = tau - tau0
            
        # Set up a low-level constructor that builds a satellite model 
        # directly from numeric orbital parameters
        # Code reference <https://rhodesmill.org/skyfield/earth-satellites.html>
        # See <https://pypi.org/project/sgp4/#providing-your-own-elements> 
        satrec.sgp4init(
            WGS72,                 # gravity model
            'i',                   # 'a' = old AFSPC mode, 'i' = improved mode
            0,                     # satnum: Satellite number (0 for custom satellites)
            epoch,                 # epoch: days since 1949 December 31 00:00 UT
            0.0,                   # bstar: drag coefficient (/earth radii)
            0.0,                   # ndot: ballistic coefficient (revs/day)
            0.0,                   # nddot: second derivative of mean motion (revs/day^3)
            ecco,                  # ecco: eccentricity
            np.radians(argpo),     # argpo: argument of perigee (radians)
            np.radians(inclo),     # inclo: inclination (radians)
            np.radians(mo),        # mo: mean anomaly (radians)
            (2 * np.pi) / period,  # no_kozai: mean motion (radians/minute)
            np.radians(RAAN),      # nodeo: right ascension of ascending node (radians)
        )
        
        # Wrap this low-level satellite model in a Skyfield EarthSatellite object
        ts = load.timescale()
        self.earth_satellite = EarthSatellite.from_satrec(satrec, ts)
        
        return self.earth_satellite

    
    def build_from_TLE(self, line1, line2):
        """
        Initialize a Skyfield EarthSatellite object from Two-Line Element (TLE) data.

        Parameters
        ----------
        line1, line2 : str
            The first and second line of the two-line element set (TLE).
        
        Returns
        -------
        skyfield.sgp4lib.EarthSatellite
             A Skyfield EarthSatellite object used to model the satellite's orbital
             trajectory and predict its position over time.
        """
        self.earth_satellite = EarthSatellite(line1, line2)

        return self.earth_satellite

        