import numpy as np
import astropy.units as u
from astropy.time import Time
from skyfield.api import load

from utilities import cart_to_equi
from targets import SolarSystemTarget

def find_CVZcenter(earth_satellite, time):
    """
    Find the equatorial coordinates (RA, Dec) corresponding to the Continuous Viewing
    Zone (CVZ) center of the satellite at the given time.

    Parameters
    ----------
    earth_satellite : skyfield.sgp4lib.EarthSatellite
        The Skyfield EarthSatellite object representing the satellite in operation.
    
    time : astropy.time.Time
        The time of observation.
    
    Returns
    -------
        CVZ_ra, CVZ_dec : float
    """
    # Define the Right Ascension of the CVZ center
    # as the RA of anti Sun direction
    t = load.timescale().from_astropy(time)
    CVZ_ra, _ = cart_to_equi(-SolarSystemTarget('sun').pointing(earth_satellite, time))
    
    # Define the Declination of the CVZ center
    inclo = np.degrees(earth_satellite.model.inclo) # orbital inclination
    CVZ_dec = 90 - inclo
    
    return CVZ_ra, CVZ_dec


def find_anti_SunRA_day(surveyRA, VE):
    """
    Finds the day when the Solar longitude (the Sun's RA position) is positioned 180° 
    away along the ecliptic from `surveyRA` on the celestial sphere; i.e., the day when 
    the survey is in the opposite direction from the Sun as seen from Earth. 
    
    Parameters
    ----------
    surveyRA : float
        The Right Ascension of the survey in degrees within the range [-180°, +180°].
    
    VE : astropy.time.Time
        The Vernal Equinox time of the current year in UTC.
        Example: VE = Time("2024-03-20 03:06:00") 
    
    Returns
    -------
    anti_SunRA_day : astropy.time.Time
        The date when the Sun's ecliptic longitude is opposite the surveyRA.
    """
    # Define the Sun's anti-ecliptic longitude (opposite of Sun's ecliptic longitude)
    anti_longitudes = np.arange(-180, 180, step=360 / 365.25)
    anti_sun_RA_data = [
        lon + 2.45 * np.sin(np.radians(2 * lon)) for lon in anti_longitudes
    ]

    # Interpolate to find the day corresponding to the given surveyRA
    delta_day_data = np.arange(len(anti_sun_RA_data))
    delta_day = np.interp(surveyRA, anti_sun_RA_data, delta_day_data)

    # Calculate the exact date
    anti_SunRA_date = VE + delta_day * u.day
    anti_SunRA_day = Time(anti_SunRA_date.iso.split()[0])  # Extract the date part
    
    return anti_SunRA_day


def find_occultation(tile, Now, duration, subexposure, period, earth_satellite):
    """ 
    Determines whether occultation occurs by checking if the `tile` is visible throughout
    the `subexposure` time, which starts `duration` after `Now`. If there is occultation, 
    it defines the visibility windows over an orbital period, and saves it as a `tile` 
    attribute. Then calculates the occulted time from `Now` to the start of the tile's
    new visible cycle.

    Parameters
    ----------
    tile : InertialTarget 
        The target tile to check for occultation.
   
    Now : astropy.time.Time
        The current time of observation.
    
    duration : float
        The time in seconds from `Now` before the planned imaging start time.
    
    subexposure : float
        The continuous open-shutter time in seconds.
    
    period : float
        Satellite's orbital period in minutes.

    earth_satellite : skyfield.sgp4lib.EarthSatellite
        The Skyfield EarthSatellite object representing the satellite in operation.

    Returns
    -------
    occultation : float
        The time in seconds when the `tile` is not visible (e.g., occulted by the 
        Earth's limb).

    Raises
    ------
    ValueError
        If the target currently has no more visibility windows. 
    """
    import timeit

    code_start = timeit.default_timer()

    if (
        tile.avoid_bright_objects(
            earth_satellite, Now + duration * u.s
        ) is True and
        tile.avoid_bright_objects(
            earth_satellite, Now + duration * u.s + subexposure * u.s
        ) is True
    ):
        # If the tile is visible throughout the open-shutter time,
        # there is no occultation.
        occultation = 0
        # print(f'occultation = {round(occultation / 60, 2)} min')
    
    else:
        # Scan for the tile's visibility window per minute over an orbital period.
        tile.orbit_vis = tile.get_visibility_windows(
            Now, Now + period * u.min, 60, earth_satellite
        )
        if tile.orbit_vis.num == 0:
            raise ValueError("The target currently has no visibility window.")

        # Calculate occultation
        # i.e., the time from `Now` to the start of the new visible cycle.
        occultation = (tile.orbit_vis.get_start_time() - Now).sec
        # print(f'occultation = {round(occultation / 60, 2)} min')

        runtime = timeit.default_timer() - code_start
        # print(f"runtime: {runtime} sec")

    if occultation < 0:
        raise ValueError("Occultation time cannot be negative!")

    return occultation