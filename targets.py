from abc import ABC, abstractmethod

from astropy.coordinates import SkyCoord, GCRS
import astropy.units as u
from astropy.time import Time
from skyfield.api import EarthSatellite, load, wgs84
import numpy as np

from windows import Window

class Target(ABC):
    """ 
    Abstract base class for all targets.
    """
    # Class attribute shared by all subclasses
    ephem = load('de421.bsp')
    
    def __init__(self):
        # Note:
        # r_obs_target: the position vector pointing from the observer to the target 
            # (in GCRS coordinate system)
        # n_obs_target: the normalized target position vector 
            # (i.e. target pointing vector)
        pass

    
    @abstractmethod
    def pointing(self, earth_satellite=None, time=None):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI)
        frame.
        
        Parameters
        ----------
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
        
        time : astropy.time.Time
            The time of observation. Used for determining the position of moving targets.
        
        Returns
        -------
        n_sat_target : numpy.ndarray
            The normalized vector [x, y, z] in the GCRS frame pointing from the satellite
            to the target.
        """
        pass

    
    def _target_bright_angle(self, bright_name, earth_satellite, time):
        """
        Calculates the angle (in degrees) between the target and the bright object
        observed from the satellite.
    
        Parameters
        ----------
        bright_name : str
            Must be one of 'sun', 'earth', or 'moon'.
    
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
    
        time : astropy.time.Time
            The time of observation. Used to determine the position of the satellite and
            bright object.
    
        Returns
        -------
        angle : float
            The angle in degrees between the target and the bright object from the 
            satellite's perspective.
        """
        if bright_name not in ['sun', 'earth', 'moon']:
            raise ValueError(
                f"Invalid 'bright_name': {bright_name}. "
                "Must be one of 'sun', 'earth', or 'moon'."
            )
    
        # GCRS position vector from satellite to the bright object
        t = load.timescale().from_astropy(time)  # convert to Skyfield time
        r_geo_sat = (
            earth_satellite.at(t)
            .to_skycoord().cartesian.xyz.to(u.km).value  # km
        )
        r_geo_bright = (
            self.ephem['earth'].at(t).observe(self.ephem[bright_name])
            .to_skycoord().cartesian.xyz.to(u.km).value  # km
        )
        r_sat_bright = r_geo_bright - r_geo_sat
    
        # Angle between bright object pointing vector and target pointing vector
        n_sat_bright = r_sat_bright / np.linalg.norm(r_sat_bright)
        n_sat_target = self.pointing(earth_satellite, time)
        angle = np.degrees(np.arccos(np.dot(n_sat_bright, n_sat_target)))
        
        return angle


    def _bright_exclusion_angle(
        self, bright_name, earth_satellite, time, SEA=10, ELEA=10, MLEA=7
    ):
        """
        Calculates the exclusion angles (in degrees) of the bright objects Sun, Earth, 
        and Moon.
    
        Parameters
        ----------
        bright_name : str
            Must be one of 'sun', 'earth', or 'moon'.
    
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
    
        time : astropy.time.Time
            The time of observation.
    
        SEA : float, optional
            The Sun Exclusion Angle (default is 10 degrees).
    
        ELEA : float, optional
            The Earth Limb Exclusion Angle (default is 10 degrees).
    
        MLEA : float, optional
            The Moon Limb Exclusion Angle (default is 7 degrees).
    
        Returns
        -------
        float
            The exclusion angle in degrees for the specified bright object.
        """
        # Convert to Skyfield time
        t = load.timescale().from_astropy(time)
        
        if bright_name == 'sun':
            return SEA
        
        elif bright_name == 'earth':
            # Earth radius
            Re = 6378.1  # km
            
            # Position vector from Earth to satellite
            r_geo_sat = (
                earth_satellite.at(t)
                .to_skycoord().cartesian.xyz.to(u.km).value  # km
            )
            
            # Distance between satellite and Earth
            dist = np.linalg.norm(r_geo_sat)  # km
            
            # Angle between Earth's center and limb (Earth Limb Angle)
            ELA = np.degrees(np.arcsin(Re / dist))  # deg
            
            # Earth Exclusion Angle
            EEA = ELA + ELEA  # deg
            
            return EEA
        
        elif bright_name == 'moon':
            # Moon radius
            Rm = 1738.1  # km
            
            # Position vector from satellite to moon
            r_geo_sat = (
                earth_satellite.at(t)
                .to_skycoord().cartesian.xyz.to(u.km).value  # km
            )
            r_geo_moon = (
                self.ephem['earth'].at(t).observe(self.ephem['moon'])
                .to_skycoord().cartesian.xyz.to(u.km).value  # km
            )
            r_sat_moon = r_geo_moon - r_geo_sat  # km
            
            # Distance between satellite and Moon
            dist = np.linalg.norm(r_sat_moon)  # km
            
            # Angle between Moon's center and limb (Moon Limb Angle)
            MLA = np.degrees(np.arcsin(Rm / dist))  # deg
            
            # Moon Exclusion Angle
            MEA = MLA + MLEA  # deg
            
            return MEA
        
        else:
            raise ValueError(
                f"Invalid 'bright_name': {bright_name}. "
                "Must be one of 'sun', 'earth', or 'moon'."
            )
    
        
    def avoid_bright_objects(self, earth_satellite, time, SEA=90, ELEA=10, MLEA=7):
        """
        Checks whether the telescope pointing direction at a given time avoids Sun, Earth,
        and Moon.
    
        Parameters
        ----------
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
        
        time : astropy.time.Time
            The time of observation.
    
        SEA : float, optional
            Sun exclusion angle with a default value of 90°.
        
        ELEA : float, optional
            Earth limb exclusion angle with a default value of 10°.
        
        MLEA : float, optional
            Moon limb exclusion angle with a default value of 7°.
    
        Returns
        -------
        bool or list of str
            Returns `True` if the separation angles between the target and the Sun, Earth,
            and Moon are all greater than their respective exclusion angles. Otherwise, 
            returns a list of object names (e.g., 'sun', 'earth', 'moon') that occult the
            target.
        """
        if isinstance(self, EarthFixedTarget):
            raise TypeError("Does not apply to earth-fixed targets.")
        
        # The separation angle between the target and the bright objects
        # seen from the satellite.
        sun_angle = self._target_bright_angle('sun', earth_satellite, time)
        earth_angle = self._target_bright_angle('earth', earth_satellite, time)
        moon_angle = self._target_bright_angle('moon', earth_satellite, time)
        
        # The bright object exclusion angles.
        SEA = self._bright_exclusion_angle('sun', earth_satellite, time, SEA=SEA)
        EEA = self._bright_exclusion_angle('earth', earth_satellite, time, ELEA=ELEA)
        MEA = self._bright_exclusion_angle('moon', earth_satellite, time, MLEA=MLEA)
        
        # The separation angle must be greater than the exclusion angle.
        if (sun_angle > SEA) and (earth_angle > EEA) and (moon_angle > MEA):
            return True
        else:
            occulter = []  # stores the object that occults the target
            if sun_angle <= SEA:
                occulter.append('sun')
            if earth_angle <= EEA:
                occulter.append('earth')
            if moon_angle <= MEA:
                occulter.append('moon')
            return occulter

    
    def get_visibility_windows(
        self, start_time, end_time, time_step, earth_satellite, SEA=90, ELEA=10, MLEA=7
    ):
        """
        Finds the time periods when the target is visible from the satellite, within the
        given time range.
    
        Parameters
        ----------
        start_time, end_time : astropy.time.Time
            The start and end time of the period for checking target visibility.
        
        time_step : int
            The increment (in seconds) to step through the checking period.
        
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
    
        SEA : float, optional
            Sun exclusion angle with a default value of 90°.
        
        ELEA : float, optional
            Earth limb exclusion angle with a default value of 10°.
        
        MLEA : float, optional
            Moon limb exclusion angle with a default value of 7°.
    
        Returns
        -------
        `Window` instance
            The visibility windows.
        """
        
        if isinstance(self, EarthFixedTarget):
            raise TypeError("Does not apply to earth-fixed targets.")
        
        # Define a time array
        time_range = np.arange(0, (end_time - start_time).to(u.s).value, time_step)
        time_array = start_time + time_range * u.s
        
        visible = []
        occulter = []
        
        # Iterates through the time array
        for time in time_array:
            # Checks if the satellite is pointing away from bright objects
            avoid_bright_result = self.avoid_bright_objects(
                earth_satellite, time, SEA, ELEA, MLEA
            )
        
            # For Earth-orbiting targets, also check sunlit_result
            if isinstance(self, EarthOrbitingTarget):
                t = load.timescale().from_astropy(time)
                # Skyfield positionlib method
                sunlit_result = earth_satellite.at(t).is_sunlit(self.ephem)  
                if avoid_bright_result is True and sunlit_result is True:
                    visible.append(True)
                else:
                    visible.append(False)
            
            # For other target types, just check avoid_bright_result
            else:
                if avoid_bright_result is True:
                    visible.append(True)
                else:
                    visible.append(False)
                    occulter += avoid_bright_result
        
        # Stores the start and end times of consecutive True sequences
        windows = []
        # Tracks the start time of a consecutive True sequence
        start = None  
        
        for i, val in enumerate(visible):
            # If the value is True and the start time is not set
            if val is True and start is None:
                # Set the start time to the corresponding time from `time_array`
                start = time_array[i]  
            # If the value is False and the start time is already set
            elif val is False and start is not None:
                # Index back to the most recent True value
                end = time_array[i - 1]  
                # Store the time window
                windows.append((start, end)) 
                # Reset the start time tracker for the next consecutive True sequence
                start = None  
        
        # If the start time is still set after iterating
        # include this visibility window
        if start is not None:
            end = time_array[-1]  # Last element
            windows.append((start, end))
        
        # if len(windows) == 0:
        #     print(f"The target is occulted by {', '.join(list(set(occulter)))}.")
        
        return Window(windows)


class InertialTarget(Target):
    """
    Distant astronomical targets (e.g. stars, galaxies).
    """
    
    def __init__(self, ra, dec):
        """
        Initializes the target with Right Ascension (RA) and Declination (DEC) 
        equatorial coordinates in degrees.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees. Range is [-180°, +180°].
        
        dec : float
            Declination in degrees. Range is [-90°, +90°].

        Attributes
        ----------
        ra : float
            Right Ascension of the target in degrees.
        
        dec : float
            Declination of the target in degrees.
        
        skycoord : SkyCoord
            The target's coordinates in the GCRS frame.

        rotation_angle : float
            The spacecraft's roll angle when observing the target. 

        mp_object : shapely.geometry.MultiPolygon
            The MultiPolygon object representing the target tile.

        q_attitude : Quaternion
            The spacecraft's attitude quaternion when observing the target.

        obs_time_limit : float or None
            The maximum continuous observation time (in seconds) available at the given
            pointing attitude before the battery is depleted. The time is infinite if 
            there are no power limitations.
        
        status : str
            "visible", "other", or "idle" indicating the target's visibility status.
        
        orbit_vis : Window 
            The target's visibility windows over an orbital period.
        
        occultation : numpy.float64
            Time (in seconds) over an orbital period where the target is occulted.
        
        subexposure_count : int
            Count of subexposures completed, starting at 0.
        
        index : int, optional
            The index indicating the target's observation order in the observation 
            sequence.
       
        timestamp : astropy.time.Time, optional
            Records the time when the target begins imaging.
        """
        super().__init__()

        # Target coordinates
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, frame="icrs", unit="deg").transform_to(GCRS)

        # Initialize other attributes
        # assigned in tiling
        self.rotation_angle = None # The spacecraft roll when observing the target
        self.mp_object = None # The MultiPolygon object representing the target tile
        # assigned in slew calculations
        self.q_attitude = None # Attitude quaternion when the target is observed
        self.net_power = None # Net power generated when pointing at the target
        # assigned in survey observations
        self.status = None
        self.orbit_vis = None
        self.occultation = None
        self.subexposure_count = 0
        self.index = None  # Index of the target in observation order
        self.timestamp = None  # Time when the target begins imaging

    def __repr__(self):
        return (
            f"<InertialTarget object: ra={self.ra:.3f}°, "
            f"dec={self.dec:.3f}°, "
            f"rotation_angle={self.rotation_angle}°, "
            f"index={self.index}, "
            f"status='{self.status}'>"
        )

    def pointing(self, earth_satellite=None, time=None):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI)
        frame.
        """
        # For distant targets, the pointing vector is approximated as the direction 
        # from Earth's center to the target.
        n_sat_target = self.skycoord.cartesian.xyz.value
        return n_sat_target


class SolarSystemTarget(Target):
    """
    Solar system objects.
    """
    
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            The name of the ephemeris targets supported by JPL ephemeris DE421.
            See <https://rhodesmill.org/skyfield/planets.html#listing-ephemeris-targets>. 
        """
        super().__init__()
        
        # Validate the ephemeris target name
        valid_names = set()

        for names_list in self.ephem.names().values():
            for n in names_list:
                # Convert to uppercase for case-insensitive comparison
                valid_names.add(n.upper()) 
        
        if name.upper() not in valid_names:
            raise ValueError(
                f"'{name}' is not a valid name in the loaded ephemeris. "
                f"Valid names include: {', '.join(sorted(valid_names))}."
            )
        
        self.name = name

    
    def pointing(self, earth_satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI)
        frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = (
            self.ephem['earth'].at(t).observe(self.ephem[self.name])
            .to_skycoord().cartesian.xyz.to(u.km).value # km
        )
        r_geo_sat = earth_satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target


class EarthOrbitingTarget(Target):
    """
    Deployed Earth satellites (moving targets).
    """
    
    def __init__(self, line1, line2):
        """
        Parameters
        ----------
        line1, line2 : str
            The first and second line of the two-line element set (TLE).
        """
        super().__init__()
        self.line1 = line1
        self.line2 = line2
        self.sat = EarthSatellite(line1, line2)

    
    def pointing(self, earth_satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) 
        frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = self.sat.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_geo_sat = earth_satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target


class EarthFixedTarget(Target):
    """
    Locations on Earth's surface.
    """
    def __init__(self, lat, lon):
        """
        Parameters
        ----------
        lat, lon : float
            The latitude range is [-90°, +90°] with North as positive.
            The longitude range is [-180°, +180°] with East as positive.
        """
        super().__init__()
        self.lat = lat
        self.lon = lon

    
    def pointing(self, earth_satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) 
        frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = (
            wgs84.latlon(self.lat, self.lon).at(t)
            .to_skycoord().cartesian.xyz.to(u.km).value # km
        )
        r_geo_sat = earth_satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target

    
    def is_clear_sky(self, time, max_cloud_cover=30, years=10):
        """
        Estimates whether the sky of the target location on a given day in the year 
        is clear enough for optical downlinking. Uses historical cloud coverage data 
        from the Open-Meteo API (https://open-meteo.com/).
        
        Parameters
        ----------
        time : astropy.time.Time
            The day of interest.
        
        max_cloud_cover : float, optional
            Maximum cloud coverage percentage allowed for optical downlinking. 
            Default is 30%.
        
        years : int, optional
            Number of past years to check historical cloud coverage. Default is 10 years.
        
        Returns
        -------
        bool
            `True` if the average cloud coverage percentage is below `max_cloud_cover`, 
            `False` otherwise.

        Raises
        ------
        ValueError
            If there is insufficient cloud coverage data.
        """
        import requests
        import datetime
    
        # Convert Astropy Time object to a standard datetime.date
        date = time.to_datetime().date()
        month, day = date.month, date.day
        
        # List to store historical cloud coverage for the same month and day
        historical_cloud_covers = []
        
        # Get current year
        current_year = datetime.date.today().year
        
        # Looping over past years to retrieve historical data
        for i in range(1, years + 1):
            year = current_year - i

            # Handle leap years for February 29th
            if (
                month == 2 and 
                day == 29 and 
                not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
            ):
                continue  # Skip non-leap years
            
            # Build the historical date string
            historical_date_str = f"{year}-{month:02d}-{day:02d}"
            
            try:
                # Make the API request to fetch cloud cover for the historical date
                response = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        'latitude': self.lat,
                        'longitude': self.lon,
                        'start_date': historical_date_str,
                        'end_date': historical_date_str,
                        'hourly': 'cloudcover',
                        'timezone': 'UTC'
                    },
                    timeout=10  # Prevent code from running too long
                )
                response.raise_for_status()
                data = response.json()
    
                # Extract cloud cover data
                cloud_cover = data['hourly']['cloudcover']
                if cloud_cover:
                    # Average hourly cloud coverage over the day
                    avg_cloud_cover = sum(cloud_cover) / len(cloud_cover)
                    historical_cloud_covers.append(avg_cloud_cover)
            
            except (requests.exceptions.RequestException, KeyError) as e:
                print(f"Data error for {year}: {e}")
        
        # Calculate the average cloud cover over the past years
        if not historical_cloud_covers:
            raise ValueError("Insufficient cloud coverage data. Try increasing `years`.")
        
        avg_cloud_cover = sum(historical_cloud_covers) / len(historical_cloud_covers)
        # print(f"Avg cloud coverage over {years} years: {avg_cloud_cover:.2f}%")
        
        # Determine if the sky is clear
        return avg_cloud_cover <= max_cloud_cover

    
    def find_periods_above_elevation(
        self, start_time, end_time, min_elevation, earth_satellite
    ):
        """
        Searches between `start_time` and `end_time` for the time periods during which 
        the satellite is above `min_elevation` degrees relative to the horizon of the 
        ground station.
        
        This function uses Skyfield's sgp4lib find_events() method, which returns a 
        tuple (t, events), where the first element is a Skyfield timelib.Time object in 
        Terrestrial Time, and the second element 
        is an array of events:
            * 0 — Satellite rises above `min_elevation`.
            * 1 — Satellite culminates and begins to descend.
            * 2 — Satellite falls below `min_elevation`.

        Parameters
        ----------
        start_time, end_time : astropy.time.Time
            The start and end time of the search period.
            
        min_elevation : float
            The minimum elevation angle in degrees.
            
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite bring tracked.

        Returns
        -------
        time_periods : list of tuples
            Each tuple represents a time period as (`rising_time`, `falling_time`)
            which are Astropy Time objects in UTC.
        """
        # Ground station location
        topos = wgs84.latlon(self.lat, self.lon)
        
        # Convert start and end times to Skyfield time scale
        t0 = load.timescale().from_astropy(start_time)
        tf = load.timescale().from_astropy(end_time)
        
        # Get the satellite events during the given time range
        t, events = earth_satellite.find_events(topos, t0, tf, min_elevation)
        
        # List to store the time periods when the satellite is above the given elevation
        time_periods = []
        rising_time = None
        
        # Loop through the events to identify rising and falling times
        for i in range(len(events)):
             # Satellite rises above the elevation
            if events[i] == 0: 
                rising_time = t[i].to_astropy()
                rising_time.format = 'iso'
            # Satellite falls below the elevation
            elif events[i] == 2 and rising_time is not None:  
                falling_time = t[i].to_astropy()
                falling_time.format = 'iso'
                # Convert from tt to utc
                time_periods.append((rising_time.utc, falling_time.utc))  
                rising_time = None
        
        # If the satellite is still above `min_elevation` at the end of the time period
        if rising_time is not None:
            falling_time = t[-1].to_astropy()
            falling_time.format = 'iso'
            # Convert from tt to utc
            time_periods.append((rising_time.utc, falling_time.utc)) 
        
        return time_periods

    
    def get_downlink_windows(
        self, start_time, end_time, min_elevation, earth_satellite, max_cloud_cover=None
    ):
        """
        Within the given `start_time` and `end_time` period, search for the time periods 
        when the satellite pass times (i.e. in contact with the ground station). The pass
        time is when the satellite is above `min_elevation` of the ground station's 
        horizon. Note that optical downlink requires the sky to be clear.
    
        Parameters
        ----------
        start_time, end_time : astropy.time.Time
            The start and end time of the search period for downlink opportunities. 

        min_elevation : float
            The minimum elevation angle in degrees.
            
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite performing
            downlink.

        max_cloud_cover : float, optional
            The maximum cloud coverage percentage allowed for optical downlinking.

        Returns
        -------
        `Window` instance
            The downlink windows.
        """
        # If the `max_cloud_cover` parameter is given
        # check whether the sky is clear on the given day
        if max_cloud_cover is not None:
            if not self.is_clear_sky(start_time, max_cloud_cover):
                return Window()  # No downlink possible due to cloudy sky
        
        # Find the time periods in which the satellite is above `min_elevation` degrees
        # above the horizon of the ground station (i.e. satellite pass time)
        passtimes = self.find_periods_above_elevation(
            start_time, end_time, min_elevation, earth_satellite
        )
        
        return Window(passtimes)