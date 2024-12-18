import numpy as np
from pyquaternion import Quaternion
import timeit

from targets import InertialTarget, SolarSystemTarget, EarthOrbitingTarget, EarthFixedTarget
from utilities import equi_to_cart, print_time
from power import check_power

def get_attitude(np_eci, np_b, ns_eci, ns_b_goal):
    """
    Defines the spacecraft attitude, represented by a rotation quaternion from 
    the Earth-Centered Inertial (ECI) frame to the body frame, using primary 
    and secondary alignments. 

    The method first aligns the primary ECI vector exactly with its corresponding 
    body frame vector. Then, it aligns the secondary ECI vector as closely as possible 
    to the desired secondary body frame vector while maintaining the primary alignment.

    Parameters
    ----------
    np_eci : numpy.ndarray
        Normalized primary vector in the ECI frame.

    np_b : numpy.ndarray
        Normalized primary vector in the body frame.

    ns_eci : numpy.ndarray
        Normalized secondary vector in the ECI frame.

    ns_b_goal : numpy.ndarray
        Normalized desired secondary vector in the body frame.

    Returns
    -------
    q_eci_to_b : Quaternion 
        A quaternion that rotates a vector from the ECI frame to the body frame, 
        considering the secondary alignment.
    """
    # Define the primary rotation quaternion
    # Primary rotation axis
    np_axis = np.cross(np_b, np_eci) / np.linalg.norm(np.cross(np_b, np_eci))

    # Primary rotation angle
    θp = np.arccos(np.dot(np_b, np_eci))

    # Create the primary rotation quaternion
    # Quaternion format: q = [w, x, y, z] = [scalar, vector]
    q_eci_to_int = Quaternion(axis=np_axis, radians=θp)

    # Apply the primary rotation to transform the secondary ECI vector 
    # into an intermediate frame
    # Active rotation: the vector is rotated with respect to the coordinate system
    ns_int = (q_eci_to_int.inverse * Quaternion(scalar=0, vector=ns_eci) * q_eci_to_int).vector

    # Define the secondary rotation quaternion
    # Secondary rotation axis is the primary body frame vector
    # Secondary rotation angle is chosen to maximize the dot product 
    # between ns_int and ns_b_goal
    A = np.dot(ns_int, ns_b_goal) - np.dot(np_b, ns_int) * np.dot(np_b, ns_b_goal)
    B = np.dot(np.cross(ns_int, np_b), ns_b_goal)
    d_s = A / np.sqrt(A**2 + B**2)

    # Scalar and vector components of the secondary rotation quaternion
    q_scal = np.sign(B) * np.sqrt((1 + d_s) / 2)
    q_vec = np_b * np.sqrt((1 - d_s) / 2)
    q_int_to_b = Quaternion(scalar=q_scal, vector=q_vec)

    # Final attitude quaternion that rotates any ECI vector into the body frame
    q_eci_to_b = q_eci_to_int * q_int_to_b
    return q_eci_to_b


def _get_cubic_slew(slew_angle, slew_duration, sc_inertia):
    """
    Computes the cubic slew profile for a given slew angle and duration. 
    The function calculates the quaternion trajectory, angular rates, angular
    accelerations, spacecraft angular momentum, and torque at discrete time steps 
    along the slew.

    Parameters
    ----------
    slew_angle : float
        The slew angle in degrees. Assumes the rotation is about the Z-axis of the
        spacecraft.

    slew_duration : float
        The time duration in seconds for the spacecraft to complete the slew.

    sc_inertia : numpy.ndarray
        A 3x3 symmetric matrix representing the spacecraft's inertia tensor 
        (units: kg·m²), describing the mass distribution along each axis.

    Returns
    -------
    t_slew : numpy.ndarray
        An array of time steps in seconds from 0 to `slew_duration`.
        
    q_slew : list of Quaternion
        The quaternion trajectory at each time step.

    w_slew : numpy.ndarray
        An Nx3 array where each row contains the angular velocity components (x, y, z) 
        at the corresponding time step (units: rad/s).

    a_slew : numpy.ndarray
        An Nx3 array where each row contains the angular acceleration components 
        (x, y, z) at the corresponding time step (units: rad/s²).

    h_slew : numpy.ndarray
        An Nx3 array where each row contains the spacecraft's angular momentum components
        (x, y, z) at the corresponding time step (units: N·m·s).

    tau_slew : numpy.ndarray
        An Nx3 array where each row contains the spacecraft's torque components (x, y, z)
        at the corresponding time step (units: N·m).
    """

    # Initialize the initial and final quaternions for the slew
    # assuming the slew rotates about the Z axis
    init_q = Quaternion(axis=[0, 0, 1], degrees=0)
    final_q = Quaternion(axis=[0, 0, 1], degrees=slew_angle)
    delta_q = final_q - init_q

    # Define initial and final quaternion derivatives
    # assuming the angular velocity is zero at endpoints
    init_q_deriv = init_q.derivative([0, 0, 0])
    final_q_deriv = final_q.derivative([0, 0, 0])
    delta_q_deriv = final_q_deriv - init_q_deriv

    # Compute the cubic polynomial coefficients for the quaternion trajectory
    # note: each coefficient variable is a Quaternion object
    c0 = init_q
    c1 = init_q_deriv
    c2 = (
        3 * (delta_q - init_q_deriv * slew_duration) - delta_q_deriv * slew_duration
    ) / (slew_duration ** 2)
    c3 = (
        -2 * (delta_q - init_q_deriv * slew_duration) + delta_q_deriv * slew_duration
    ) / (slew_duration ** 3)

    # Initialize time steps for the slew trajectory
    dt_slew = 1  # Time step size in seconds
    t_slew = np.arange(0, slew_duration + dt_slew, dt_slew)
    n_slew = len(t_slew)

    # Initialize arrays to store the trajectory data
    q_slew = [Quaternion([1, 0, 0, 0])] * n_slew  # Identity quaternion as a placeholder
    q_slew_dot = [Quaternion([0, 0, 0, 0])] * n_slew  # Zero first derivatives
    q_slew_dot_dot = [Quaternion([0, 0, 0, 0])] * n_slew  # Zero second derivatives
    w_slew = np.zeros((n_slew, 3))  # Angular rates (vector form)
    a_slew = np.zeros((n_slew, 3))  # Angular accelerations (vector form)
    h_slew = np.zeros((n_slew, 3))  # Spacecraft angular momentum
    tau_slew = np.zeros((n_slew, 3))  # Spacecraft torque

    # Compute the slew trajectory at each time step
    for j, t in enumerate(t_slew):
        # Compute the quaternion components using the cubic polynomial
        q_array = np.zeros(4)
        for i in range(4):
            q_array[i] = c0[i] + t * (c1[i] + t * (c2[i] + t * c3[i]))
        q_slew[j] = Quaternion(q_array).normalised  # Normalize the quaternion

        # Compute quaternion derivatives
        if j > 0:
            # First derivative
            q_slew_dot[j] = (q_slew[j] - q_slew[j - 1]) / dt_slew 
            # Second derivative
            q_slew_dot_dot[j] = (q_slew_dot[j] - q_slew_dot[j - 1]) / dt_slew 

        # Compute angular rate and acceleration (skip the first time step)
        if j > 0:
            w_slew[j] = 2 * (q_slew[j].conjugate * q_slew_dot[j]).vector
            a_slew[j] = 2 * (q_slew_dot[j].conjugate * q_slew_dot[j] + 
                             q_slew[j].conjugate * q_slew_dot_dot[j]).vector

            # Compute angular momentum and torque
            h_slew[j] = np.dot(sc_inertia, w_slew[j])
            tau_slew[j] = np.dot(sc_inertia, a_slew[j])

    return t_slew, q_slew, w_slew, a_slew, h_slew, tau_slew


def _check_slew(h_slew, tau_slew, max_wheel_momentum, max_wheel_torque, wheel_design_margin):
    """
    Check if the spacecraft's angular momentum and torque throughout the slew are within
    the limitations of the reaction wheel's design specifications. The function accounts
    for the momentum and torque capacity of a single wheel and applies a margin fraction
    to it.

    Parameters
    ----------
    h_slew : numpy.ndarray
        An Nx3 array where each row contains the spacecraft's angular momentum components
        (x, y, z) at the corresponding time step (units: N·m·s).

    tau_slew : numpy.ndarray
        An Nx3 array where each row contains the spacecraft's torque components (x, y, z)
        at the corresponding time step (units: N·m).
        
    max_wheel_momentum : float
        The maximum allowable angular momentum of the reaction wheel 
        (units: N·m·s).

    max_wheel_torque : float
        The maximum allowable torque of the reaction wheel (units: N·m).

    wheel_design_margin : float
        The design margin factor applied to the reaction wheel's performance limits 
        as a buffer to ensure safe operations.

    Returns
    -------
    bool
        Returns `True` if both the spacecraft's angular momentum and torque are 
        within the reaction wheel's limitations, considering the design margin. 
        Returns `False` otherwise.
    """
    # Calculate the maximum magnitudes of angular momentum and torque
    max_h_magnitude = np.max(np.linalg.norm(h_slew, axis=1))
    max_tau_magnitude = np.max(np.linalg.norm(tau_slew, axis=1))
    
    # Check if the angular momentum magnitude is within the reaction wheel's limitation.
    max_h_magnitude = np.max(np.linalg.norm(h_slew, axis=1))
    h_check = max_h_magnitude < (max_wheel_momentum / wheel_design_margin)
    # print(f"Maximum angular momentum during the slew: {max_h_magnitude:.3f} N·m·s")
    # if not h_check:
    #     print(
    #         f"Angular momentum exceeds the wheel's limitation of "
    #         f"{(max_wheel_momentum / wheel_design_margin):.3f} N·m·s.")
    #     print("Consider increasing the slew duration to reduce angular momentum.")

    # Check if the torque vector magnitude is within the reaction wheel's limitation.
    max_tau_magnitude = np.max(np.linalg.norm(tau_slew, axis=1))
    tau_check = max_tau_magnitude < (max_wheel_torque / wheel_design_margin)
    # print(f"Maximum torque during the slew: {max_tau_magnitude:.3f} N·m")
    # if not tau_check:
    #     print(f"Torque exceeds the wheel's limitation of "
    #           f"{(max_wheel_torque / wheel_design_margin):.3f} N·m.")
    #     print("Consider increasing the slew duration to reduce torque.")

    return h_check and tau_check


def get_slew_data(sc_inertia, max_wheel_momentum, max_wheel_torque, wheel_design_margin):
    """
    Calculate the minimum slew durations required for slews rotating about Z-axis with
    angles ranging from 0° to 180°, ensuring the spacecraft's angular momentum and torque
    doesn't exceed the reacton wheel's limits.
    
    This function iteratively determines the shortest possible slew duration for each
    angle. Starting with a baseline duration, it incrementally evaluates whether the slew
    adheres to the reaction wheel's performance constraints and adjusts the duration 
    accordingly. For each angle, the calculated minimum duration is stored and used as
    the initial value for the subsequent angle.

    Parameters
    ----------
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

    Returns
    -------
    slew_angle_data : list of float
        A list containing the slew angles (in degrees) for which the minimum 
        durations were calculated.

    slew_time_data : list of float
        A list containing the minimum slew durations (in seconds) corresponding 
        to each slew angle in `slew_angle_data`.
    """
    print(
        "Calculating slew durations based on spacecraft's reaction wheel limitations..."
    )
    code_start = timeit.default_timer()
    
    slew_angle_data = np.arange(0.1, 180, 0.2).tolist()  # Slew angles (in degrees)
    slew_time_data = []  # To store the minimum slew duration for each angle
    delta_t = 1  # Increment duration by this value (in seconds)

    # Initialize the starting slew duration (in seconds)
    prev_slew_duration = 5

    for slew_angle in slew_angle_data:
        slew_duration = prev_slew_duration
        # print(f"\033[1mEvaluating slew angle = {slew_angle:.1f}°\033[0m")

        while True:
            # Compute the slew profile
            t_slew, q_slew, w_slew, a_slew, h_slew, tau_slew = _get_cubic_slew(
                slew_angle, slew_duration, sc_inertia
            )

            # Check if the slew satisfies spacecraft limitations
            if _check_slew(
                h_slew, tau_slew, max_wheel_momentum, 
                max_wheel_torque, wheel_design_margin
            ):
                # print(f"  \u2713 Minimum duration found: {slew_duration} seconds")
                slew_time_data.append(slew_duration)
                # Update the starting duration for the next angle
                prev_slew_duration = slew_duration 
                break

            # Increment duration and re-evaluate
            slew_duration += delta_t
            # print(f"  \u2717 Slew duration = {slew_duration - delta_t} seconds failed.")
    
    runtime = timeit.default_timer() - code_start
    print("runtime:", print_time(runtime))
    
    return slew_angle_data, slew_time_data


def field_slew(
    initial_target, final_target, earth_satellite, time, telescope_boresight, solar_array,
    solar_panel_normals, solar_panel_areas, solar_panel_eff, battery_power, power_load, 
    slew_angle_data, slew_time_data
):
    """
    Calculate the minimum time required to slew from one survey field to another.

    This function assumes all survey fields are inertial targets. The pointing quaternion 
    is determined by aligning the telescope boresight to the target and orienting the
    solar array normal as closely as possible towards the Sun. It computes the initial
    and final pointing quaternions and assigns them to the `initial_target` and 
    `final_target` attributes, respectively. 

    It evaluates whether the final target's pointing attitude results in a power deficit,
    calculates its maximum continuous observation time before the battery onboard is 
    depleted, and assigns this value to the `final_target` attribute.
    
    It calculates the minimum slew duration required, assuming the slew angle between 
    the initial and final pointing quaternions corresponds to a rotation about the 
    Z-axis only.

    Parameters
    ----------
    initial_target : InertialTarget
        A tile in the current field of observation.
    
    final_target : InertialTarget
        A tile in the field of destination.

    earth_satellite : skyfield.sgp4lib.EarthSatellite
        The Skyfield EarthSatellite object representing the satellite in operation.
        
    time : astropy.time.Time
        The approximate time of the slew, used to determine the Sun's position relative
        to the satellite in the ECI frame.

    telescope_boresight : numpy.ndarray
        A unit vector representing the telescope boresight direction in the spacecraft 
        body frame.

    solar_array : numpy.ndarray
        A unit vector representing the average direction of all antenna normals in the
        spacecraft body frame.

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

    slew_angle_data : list of float
        A list of slew angles (in degrees) for which the minimum durations were 
        pre-calculated, based on reaction wheel limitations.

    slew_time_data : list of float
        A list of minimum slew durations (in seconds) corresponding to each slew angle in
        `slew_angle_data`, based on reaction wheel limitations.

    Returns
    -------
    slew_time : float
        The minimum slew duration (in seconds) required to slew from the current field 
        to another.
    """
    # Validate target types
    if not isinstance(initial_target, InertialTarget):
        raise TypeError("`initial_target` must be an instance of InertialTarget.")
    if not isinstance(final_target, InertialTarget):
        raise TypeError("`final_target` must be an instance of InertialTarget.")

    # Compute the Sun's position in the ECI frame
    sun_pointing = SolarSystemTarget('sun').pointing(earth_satellite, time)

    # Compute the initial pointing quaternion
    initial_target.q_attitude = get_attitude(
        np_eci=initial_target.pointing(), np_b=telescope_boresight,
        ns_eci=sun_pointing, ns_b_goal=solar_array
    )

    # Compute the final pointing quaternion
    final_target.q_attitude = get_attitude(
        np_eci=final_target.pointing(), np_b=telescope_boresight,
        ns_eci=sun_pointing, ns_b_goal=solar_array
    )

    # Evaluate power constraints for the final target
    final_target.obs_time_limit = check_power(
        final_target.q_attitude, earth_satellite, time, solar_panel_normals,
        solar_panel_areas, solar_panel_eff, battery_power, power_load
    )

    # Compute the error quaternion (difference between the two pointing quaternions)
    q_error = initial_target.q_attitude.inverse * final_target.q_attitude

    # Extract the slew angle in degrees
    slew_angle = abs(q_error.degrees)

    # Interpolate to determine the corresponding slew duration
    slew_time = np.interp(slew_angle, slew_angle_data, slew_time_data)

    return slew_time
    

def _tile_orientation(coord, rotation_angle, telescope_boresight):
    """
    Defines the telescope pointing quaternion (i.e. spacecraft roll) to match 
    with the given tile's orientation (i.e. `rotation_angle`). 
    
    Parameters
    ----------
    coord : tuple 
        Tile center coordinates (RA, DEC) in degrees.
    
    rotation_angle: float
        Tile rotation angle in degrees counter-clockwise.
    
    Returns
    -------
    q_attitude : Quaternion
        The pointing quaternion.
    """
    # Optical Axis unit vector: points toward the tile's center coordinate
    u_oa = equi_to_cart(coord)
    
    # North Celestial Pole unit vector: points towards the NCP 
    # but perpendicular to the Optical Axis.
    # define the actual north celestial pole vector
    NCP = [0, 0, 1] 
    # define the rotation axis unit vector
    k = np.cross(u_oa, NCP) / np.linalg.norm(np.cross(u_oa, NCP))
    # rotate u_oa by 90° around axis k (see Rodrigues' rotation formula)
    u_ncp = np.cross(k, u_oa)

    # Tile Rotation unit vector
    # Defined by rotating the NCP Vector about the Optical Axis Vector 
    # by the rotation_angle (see Rodrigues' rotation formula)
    u_tr = (
        u_ncp * np.cos(rotation_angle) +
        np.cross(u_oa, u_ncp) * np.sin(rotation_angle) +
        u_oa * np.dot(u_oa, u_ncp) * (1 - np.cos(rotation_angle))
    )
    
    # Pointing quaternion
    # Defined by rotating the telescope boresight to Optical Axis Vector (primary) 
    # and +Z body to Tile Rotation Vector (secondary)
    # Note: assume +Z body is in the same direction as pointing from center 
    # to the top of the tile
    q_attitude = get_attitude(u_oa, telescope_boresight, u_tr, np.array([0,0,1]))
    
    return q_attitude


def tile_slew(
    initial_tile, final_tile, rotation_angle, telescope_boresight, earth_satellite,
    time, solar_panel_normals, solar_panel_areas, solar_panel_eff, battery_power, 
    power_load, slew_angle_data, slew_time_data
):
    """
    Calculate the minimum time required to slew from one tile to another tile in a survey 
    field.

    This function assumes all tiles are inertial targets. The pointing quaternion of each
    tile is computed to align with the tile's orientation (`rotation_angle`) and is
    assigned as an attribute of the respective tile. The function evaluates whether the 
    pointing attitude of the final tile results in a power deficit, calculates its 
    maximum continuous observation time before the battery onboard is depleted, and 
    assigns this value as an attribute of `final_tile`.

    It calculates the minimum slew duration required, assuming the slew angle between the 
    initial and final pointing quaternions corresponds to a rotation about the Z-axis
    only.

    Parameters
    ----------
    initial_tile : InertialTarget
        The tile currently under observation.

    final_tile : InertialTarget
        The tile to be observed following the slew.

    rotation_angle : float
        The rotation angle of the tile in degrees, measured counterclockwise.

    telescope_boresight : numpy.ndarray
        A unit vector representing the telescope boresight direction in the spacecraft
        body frame.

    earth_satellite : skyfield.sgp4lib.EarthSatellite
        The Skyfield EarthSatellite object representing the satellite in operation.

    time : astropy.time.Time
        The observation time, used to determine the Sun's position relative to the 
        satellite.

    solar_panel_normals : list of numpy.ndarray
        A list of unit vectors representing the normal directions of the solar panels in
        the body frame.

    solar_panel_areas : numpy.ndarray
        A 1D array representing the area (in m²) of each solar panel, corresponding to
        `solar_panel_normals`.

    solar_panel_eff : float
        The end-of-life (EOL) efficiency of the solar panels, expressed as a decimal.

    battery_power : float
        The available battery capacity (in Watt-hours) to sustain operations.

    power_load : float
        The orbit-average power consumption (in Watts).

    slew_angle_data : list of float
        A list of slew angles (in degrees) for which the minimum durations were 
        pre-calculated, based on reaction wheel limitations.

    slew_time_data : list of float
        A list of minimum slew durations (in seconds) corresponding to each slew angle in
        `slew_angle_data`, based on reaction wheel limitations.

    Returns
    -------
    slew_time : float
        The minimum slew duration (in seconds) required to slew from the current tile to
        another.
    """
    # Validate target types
    if not isinstance(initial_tile, InertialTarget):
        raise TypeError("`initial_tile` must be an instance of InertialTarget.")
    if not isinstance(final_tile, InertialTarget):
        raise TypeError("`final_tile` must be an instance of InertialTarget.")

    # Compute the initial pointing quaternion
    initial_tile.q_attitude = _tile_orientation(
        (initial_tile.ra, initial_tile.dec), rotation_angle, telescope_boresight
    )

    # Compute the final pointing quaternion
    final_tile.q_attitude = _tile_orientation(
        (final_tile.ra, final_tile.dec), rotation_angle, telescope_boresight
    )

    # Evaluate power constraints for the final tile
    final_tile.obs_time_limit = check_power(
        final_tile.q_attitude, earth_satellite, time, solar_panel_normals,
        solar_panel_areas, solar_panel_eff, battery_power, power_load
    )

    # Compute the error quaternion (difference between the two pointing quaternions)
    q_error = initial_tile.q_attitude.inverse * final_tile.q_attitude

    # Extract the slew angle in degrees
    slew_angle = abs(q_error.degrees)

    # Interpolate to determine the corresponding slew duration
    slew_time = np.interp(slew_angle, slew_angle_data, slew_time_data)

    return slew_time