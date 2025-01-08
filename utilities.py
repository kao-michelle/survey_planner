import numpy as np
from pyquaternion import Quaternion
import math

def equi_to_cart(coord):
    """ 
    Convert equatorial coordinates to a Cartesian vector, both in the ECI frame. 

    Parameters
    ----------
    coord : tuple
        Equatorial coordinates in the form (RA, DEC), where RA and DEC are in degrees.

    Returns
    -------
    vector : numpy.ndarray
        Cartesian vector in the form [x, y, z] in the ECI frame.
    """
    alpha, delta = np.radians(coord[0]), np.radians(coord[1])
    x = np.cos(alpha) * np.cos(delta)
    y = np.sin(alpha) * np.cos(delta)
    z = np.sin(delta)
    vector = np.array([x, y, z])
    return vector


def cart_to_equi(vector):
    """
    Convert a Cartesian vector to equatorial coordinates, both in the ECI frame. 

    Parameters
    ----------
    vector : numpy.ndarray
        Cartesian vector in the form [x, y, z] in the ECI frame.

    Returns
    -------
    coord : tuple
        Equatorial coordinates (RA, DEC) in degrees.
    """
    x, y, z = vector[0], vector[1], vector[2]
    ra = np.rad2deg(math.atan2(y, x))  # already in [-180°, 180°] range
    dec = np.rad2deg(math.asin(z))
    coord = (ra, dec)
    return coord


def hms_to_deg(hr, min, sec):
    """
    Convert Right Ascension from HMS (hours, minutes, seconds) J2000 format to degrees 
    in the range [-180°, 180°].
    """
    degrees = (hr + min / 60 + sec / 3600) * 15
    while degrees >= 180:
        degrees -= 360
    while degrees < -180:
        degrees += 360
    return degrees


def dms_to_deg(deg, arcmin, arcsec):
    """
    Convert Declination from DMS (degrees, arcminutes, arcseconds) J2000 format to degrees.
    """
    if deg < 0:
        degrees = -(abs(deg) + arcmin / 60 + arcsec / 3600)
    else:
        degrees = deg + arcmin / 60 + arcsec / 3600
    return degrees


def print_time(seconds):
    """
    Convert a duration in seconds to a string in the format:
    <days>d <hours>h <minutes>m <seconds>.sss
    """
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    # Round seconds to 3 decimal places
    seconds = round(seconds, 3)

    # Handle rounding up to 60 seconds
    if seconds == 60.000:
        seconds = 0
        minutes += 1
    
    # Handle carry-over for minutes
    if minutes >= 60:
        minutes = 0
        hours += 1
    
    # Handle carry-over for hours
    if hours >= 24:
        hours = 0
        days += 1
        
    # Only show non-zero values to keep it clean
    time_str = []
    if days > 0:
        time_str.append(f"{int(days)}d")
    if hours > 0 or days > 0:
        time_str.append(f"{int(hours):02}h")
    if minutes > 0 or hours > 0 or days > 0:
        time_str.append(f"{int(minutes):02}m")
    time_str.append(f"{seconds:.3f}s")
    
    return ' '.join(time_str)