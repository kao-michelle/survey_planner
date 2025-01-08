from astropy.time import Time
import astropy.units as u

from utilities import print_time

def downlinking(
    check_period, Now, Data_Onboard, data_tracker, downlink_windows, data_threshold,
    onboard_data_cap, downlink_rate, min_downlink_time
):
    """
    Schedules data downlinking for a satellite.

    The function checks if the satellite passes over ground stations during the 
    specified `check_period` in 30-second intervals. It schedules data downlinks 
    when the following conditions are met:
    1. The satellite is currently passing over a ground station.
    2. `Data_Onboard` reaches or exceeds `data_threshold`.
    3. The available downlink time is greater than `min_downlink_time`.

    The function assumes that all downlink operations can occur simultaneously 
    with other observation tasks, except for slewing. It updates the amount of 
    data onboard, the list of remaining downlink windows, and the `data_tracker` 
    after each successful downlink.

    Parameters
    ----------
    check_period : float
        The time period (in seconds) during which downlinking may occur.
        This period does not overlap with planned slewing times.

    Now : astropy.time.Time
        The current time.

    Data_Onboard : float
        The amount of data (in GB) currently stored onboard.

    data_tracker : dict
        A tracker that records the onboard data (in GB) over time. 
        Formatted as {"time": [], "data_onboard": []}.

    downlink_windows : Window
        The available downlink windows for the current day.

    data_threshold : float
        The onboard data threshold (in GB) required to initiate a downlink.

    onboard_data_cap : float
        The maximum onboard data capacity (in GB).

    downlink_rate : float
        The rate of data downlinking (in GBps).

    min_downlink_time : int
        The minimum time (in seconds) required to initiate a downlink.

    Returns
    -------
    Data_Onboard : float
        The updated onboard data amount (in GB) after downlinking.

    Raises
    ------
    ValueError
        If the onboard data exceeds the storage capacity.
    """
    # Document the amount of data currently onboard
    data_tracker["time"].append(Now.iso)
    data_tracker["data_onboard"].append(Data_Onboard)

    if Data_Onboard > onboard_data_cap:
        raise ValueError(
            f"The current onboard data ({Data_Onboard:.2f} GB) "
            f"exceeds the storage capacity ({onboard_data_cap:.2f} GB). "
            "Consider choosing different ground stations or transmission bands."
        )

    # No downlink window available
    if downlink_windows.num == 0:
        return Data_Onboard

    # Checks every 30 seconds from `Now` until the end of `check_period`
    time_step = 30  # seconds
    check_end = Now + check_period * u.s
    for dt in range(0, int(check_period), time_step):
        now = Now + dt * u.s

        # Ensure `now` is before the earliest downlink window
        downlink_windows.list = [
            window for window in downlink_windows.list if window[1] > now
        ]
        downlink_windows.update_num()

        if downlink_windows.num == 0:
            return Data_Onboard

        # Get the earliest downlink window
        w_start, w_end = downlink_windows.list[0]

        # Determine the available downlink time
        downlink_time = min((w_end - now).sec, (check_end - now).sec)

        # Downlink occurs when:
        # (1) `now` is within the downlink window period
        # (2) onboard data exceeds the threshold
        # (3) the available `downlink_time` is sufficient
        if (w_start <= now <= w_end and
            Data_Onboard >= data_threshold and
            downlink_time >= min_downlink_time):

            # Downlink the data onboard
            data_downlinked = min(downlink_rate * downlink_time, Data_Onboard)
            Data_Onboard -= data_downlinked

            # Update downlink windows to reflect the remaining ones
            downlink_windows.list.pop(0)
            downlink_windows.update_num()

            # print(f"\033[1mDownlinking at {now}\033[0m")
            # print(f"spends {print_time(data_downlinked / downlink_rate)} "
            #       f"of the {print_time(downlink_time)} total pass time downlinking")
            # print(f"Data downlinked: {data_downlinked:.2f} GB")
            # print(f"Remaining data onboard: {Data_Onboard:.2f} GB")

            # Document the amount of data onboard after downlinking
            data_tracker["time"].append(now.iso)
            data_tracker["data_onboard"].append(Data_Onboard)

    return Data_Onboard
