from pyquaternion import Quaternion
import numpy as np

from targets import SolarSystemTarget

def check_power(
    q_attitude, earth_satellite, time, solar_panel_normals, solar_panel_areas, 
    solar_panel_eff, battery_power, power_load
):
    """
    Calculates whether the satellite's current attitude results in a power deficit and 
    determines the maximum continuous observation time when consuming the battery onboard.

    Parameters
    ----------
    q_attitude : Quaternion object
        A quaternion representing the current spacecraft attitude. 
        It rotates a vector from the ECI frame to the body frame.

    earth_satellite : skyfield.sgp4lib.EarthSatellite
        The Skyfield EarthSatellite object representing the satellite in operation.

    time : astropy.time.Time
        The observation time, used to determine the Sun's position 
        relative to the satellite.

    solar_panel_normals : list of numpy.ndarray
        A list of unit vectors representing the normal directions of the solar arrays in
        the body frame.

    solar_panel_areas : numpy.ndarray
        A 1D array representing the area (in m²) of each solar array, corresponding to 
        `solar_panel_normals`.

    solar_panel_eff : float
        End-of-life (EOL) efficiency of the solar panels, expressed as a decimal.

    battery_power : float
        The available battery capacity (in Watt-hours) to sustain operations.

    power_load : float
        The orbit-average power consumption (in Watts).

    Returns
    -------
    obs_time_limit : float
        The maximum continuous observation time (in seconds) available at the given
        attitude before the battery is depleted. The time is infinite if there are 
        no power limitations.
    """
    # Define the Sun direction vector in the ECI frame
    sun_direction_eci = SolarSystemTarget('sun').pointing(earth_satellite, time)

    # Rotate the Sun direction vector from the ECI frame to the body frame
    sun_direction_body = (
        q_attitude.inverse * Quaternion(scalar=0, vector=sun_direction_eci) * q_attitude
    ).vector

    # Compute the cosine of the angle between the Sun vector and each solar panel normal
    # Set values to zero for panels facing away from the Sun
    panel_cosines = np.dot(solar_panel_normals, sun_direction_body)
    panel_cosines[panel_cosines < 0] = 0

    # Calculate the total power input (in Watts) from all solar arrays
    solar_flux = 1367  # Solar constant in W/m²
    panel_power_input = np.sum(
        solar_flux * solar_panel_eff * solar_panel_areas * panel_cosines
    )
    # print(f"The power generation after slew is {panel_power_input:.0f} W.")

    # Determine if the spacecraft is in a power deficit
    if power_load > panel_power_input:
        # Calculate the maximum observation time (in seconds) 
        # before the battery is depleted
        obs_time_limit = (battery_power / (power_load - panel_power_input)) * 3600
    else:
        # No observation time limitations if power generation exceeds consumption
        obs_time_limit = np.inf

    return obs_time_limit