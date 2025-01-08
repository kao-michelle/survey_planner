from abc import ABC, abstractmethod

import numpy as np
import astropy.units as u
from astropy.time import Time
from skyfield.api import load
from shapely.geometry import Polygon
import timeit
import copy

from tiling import Tiling
from tiling import _rotate_move_grid, cut_footprint, snake_scan_grid, orient_tiles, bin_tiles
from utilities import cart_to_equi, print_time
from targets import InertialTarget, SolarSystemTarget, EarthOrbitingTarget, EarthFixedTarget
from attitude import get_attitude, optimal_roll_angle, tile_pointing_attitude, perform_slew
import parameters as params
from observation import find_CVZcenter, find_anti_SunRA_day, find_occultation
from downlink import downlinking
from windows import Window
from power import update_battery

class Survey(ABC):
    """
    Abstract base class for all surveys.
    """
    
    def __init__(self):
        pass
        

class SmallSurvey(Survey):
    """
    A small survey typically covers an area around 100 square degrees or less, designed
    to have multiple visits within an observing season. In other words, the exposures are 
    repeated a large number of times at regular intervals, such as a daily cadence 
    survey.  
    """
    
    def __init__(
        self,
        satellite,
        boundary_coords, 
        exposure, 
        downlink_freq,
        subexposure_num=params.DITHER_POINT_NUM, 
        settle=params.SETTLE_TIME, 
        acquire_guide=params.GUIDE_STAR_ACQUISITION_TIME,
        dither=params.DITHER_TIME, 
        readout=params.READOUT_TIME, 
        data=params.DATA,
        data_compression_fac=params.DATA_COMPRESSION_FAC,
        internal_data_rate=params.INTERNAL_DATA_RATE,
        onboard_data_cap=params.ONBOARD_DATA_CAP,
        downlink_rate_kwargs=params.DOWNLINK_RATE,
        min_elevation_kwargs=params.MIN_ELEVATION,
        max_cloud_cover_kwargs=params.MAX_CLOUD_COVER,
        ground_stations_kwargs=params.GROUND_STATIONS,
        solar_panel_normals=params.SOLAR_PANEL_NORMALS,
        solar_panel_areas=params.SOLAR_PANEL_AREAS,
        solar_panel_eff=params.SOLAR_PANEL_EFF,
        battery_cap=params.BATTERY_CAP,
        battery_dod=params.BATTERY_DOD,
        power_load=params.POWER_LOAD,
    ):
        """
        Create a SmallSurvey instance.
        All parameters are stored as attributes.
        
        Parameters
        ----------
        satellite : Satellite
            The satellite observing the survey.
        
        boundary_coords: np.ndarray
            An array of survey footprint's boundary coordinates as (ra, dec) in degrees.

        exposure : float
            The exoposure (or open-shutter) time per tile in seconds. 

        downlink_freq : str
            The frequency band selected for data downlink (e.g., "X", "Ka", or "optical").

        subexposure_num : int
            Number of dither points to improve sub-pixel sampling.

        settle : float
            The time, in seconds, for the spacecraft to settle after slewing
            and for the Fine Guidance Sensor (FGS) to scan for targets.

        acquire_guide : float
            The time, in seconds, for the FGS to lock onto guide stars.

        dither : float
            The total time, in seconds, to perform the dither pattern for each tile. 

        readout : float
            Data readout time in seconds.

        data : float
            The amount of data, in GB, obtained per subexposure readout. 
        
        data_compression_fac : int
            The onboard data compression factor.

        internal_data_rate : float
            The internal data generation rate, in GBps, due to operations. 

        onboard_data_cap : float
            The onboard data capacity in GB.

        downlink_rate_kwargs : dict
            Dictionary specifying the data downlink rate (in GBps) for each frequency 
            band.

        min_elevation_kwargs : dict
            Dictionary defining the minimum satellite elevation angle (in degrees) 
            required to establish contact with ground stations for each frequency band.

        max_cloud_cover_kwargs : dict
            Dictionary specifying the maximum cloud coverage (in percentage) permitted 
            for optical downlinking. Set to `None` for non-optical frequency bands.

        ground_stations_kwargs : dict
            Dictionary containing lists of ground station coordinates 
            (latitude, longitude) available for each frequency band.

        solar_panel_normals : list of numpy.ndarray
            A list of unit vectors representing the normal directions of the solar arrays 
            in the spacecraft body frame.
    
        solar_panel_areas : numpy.ndarray
            A 1D array representing the area (in m²) of each solar array, 
            corresponding to `solar_panel_normals`.
    
        solar_panel_eff : float
            End-of-life (EOL) efficiency of the solar panels, expressed as a decimal.
    
        battery_cap : float
            The total energy capacity (in Watt-hours) of the battery. 

        battery_dod : float
            The allowable depth-of-discharge limit for the battery, expressed as a decimal
            fraction of its total capacity.
    
        power_load : float
            The orbit-average power consumption (in Watts).

        Attributes
        ----------
        downlink_rate : float
            The data downlink rate, in GBps, of the chosen frequency band.

        min_elevation : float
            The minimum satellite elevation, in degrees, to establish contact with the 
            ground stations of the chosen frequency band.

        max_cloud_cover : float or None
            The maximum percentage of cloud coverage that still allows for optical 
            downlinking. `None` for non-optical frequency bands.

        ground_stations : list of tuples
            A list of ground station coordinates (latitude, longitude) available for the
            chosen frequency band.

        tile_targets : list of InertialTarget
            Stores the tiles as InertialTarget in the snake scan sequence.

        num_days : int
            The number of days to observe the survey.

        Returns
        -------
        `SmallSurvey` instance
        """
        super().__init__()

        # Satellite object
        self.satellite = satellite
        if satellite.earth_satellite is None:
            raise TypeError(
                "The Skyfield EarthSatellite object is not defined. "
                "The `build_from_params()` or `build_from_TLE()` method "
                "must be called on the Satellite instance first."
            )

        # Survey footprint boundary
        self.boundary_coords = boundary_coords
        
        # Imaging specifications
        self.exposure = exposure # exposure time per tile 
        self.subexposure_num = subexposure_num # depends on the dither pattern
       
        # Opertion task times (in seconds)
        self.settle = settle # settle time + FGS(Fine Guidance Sensors) scan ID time
        self.acquire_guide = acquire_guide # guide star acquisition time
        self.dither = dither # an estimate of total (primary + subpx) dither time per tile
        self.readout = readout # read out time per subexposure
        
        # Data 
        self.data = data # amount of data (in GB) obtained per subexposure readout
        self.data_compression_fac = data_compression_fac # onboard data compression factor
        self.internal_data_rate = internal_data_rate # internal generation rate (in GBps)
        self.onboard_data_cap = onboard_data_cap # onboard data capacity (in GB)

        # Downlink parameters
        self.downlink_rate_kwargs = downlink_rate_kwargs
        self.min_elevation_kwargs = min_elevation_kwargs
        self.max_cloud_cover_kwargs = max_cloud_cover_kwargs
        self.ground_stations_kwargs = ground_stations_kwargs
        # Set downlink frequency
        self.downlink_freq = downlink_freq  # Uses the setter to initialize

        # Power
        self.solar_panel_normals = solar_panel_normals # each solar panel normal vectors
        self.solar_panel_areas = solar_panel_areas # each solar panel area (in m²)
        self.solar_panel_eff = solar_panel_eff # end-of-life efficiency 
        self.battery_cap = battery_cap # battery total capacity (in W-hr)
        self.battery_dod = battery_dod # battery depth-of-discharge limit
        self.power_load = power_load # average power consumption (in W)
        
        # Initialize potential attributes
        self.tile_targets = None # stores list of tile Target objects in tiling sequence
        self.num_days = None # number of days to observe the survey
    
    @property
    def exposure(self):
        """
        Get the current exposure time per tile.
        """
        return self._exposure

    
    @exposure.setter
    def exposure(self, value):
        """
        Set the exposure per tile to `value`.
        """
        self._exposure = value

    
    @property
    def subexposure_num(self):
        """
        Get the number of subexposures per tile 
        (which depends on the dither pattern).
        """
        return self._subexposure_num

    
    @subexposure_num.setter
    def subexposure_num(self, value):
        """
        Set the number of subexposures per tile to `value`. 
        """
        self._subexposure_num = value

    
    @property
    def subexposure(self):
        """
        Calculate the exposure time per subexposure based on
        the set `exposure` and `subexposure_num` attributes.
        """
        return self._exposure / self._subexposure_num

    
    @property
    def downlink_freq(self):
        """
        Get the current downlink frequency band.
        """
        return self._downlink_freq

    
    @downlink_freq.setter
    def downlink_freq(self, new_freq):
        """
        Set the downlink frequency band and update dependent attributes.
        """
        if new_freq not in self.downlink_rate_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the downlink_rate_kwargs dictionary."
            )
            
        if new_freq not in self.min_elevation_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the min_elevation_kwargs dictionary."
            )
        if new_freq not in self.max_cloud_cover_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the max_cloud_cover_kwargs dictionary."
            )
        if new_freq not in self.ground_stations_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the ground_stations_kwargs dictionary."
            )
        
        # Set `downlink_freq`
        self._downlink_freq = new_freq
        
        # Update dependent attributes
        # data downlink rate in GBps
        self.downlink_rate = self.downlink_rate_kwargs[new_freq]
        # satellite minimum elevation to link with ground station
        self.min_elevation = self.min_elevation_kwargs[new_freq]
        # maximum cloud coverage % for optical downlink
        self.max_cloud_cover = self.max_cloud_cover_kwargs[new_freq]
        # list of ground station coordinates 
        self.ground_stations = self.ground_stations_kwargs[new_freq]

    
    def __repr__(self):
        return (
            f"<SmallSurvey object: exposure={self.exposure} s, "
            f"downlink_freq='{self.downlink_freq}'>"
        )

    
    def _footprint_is_visible(self, candidate_day, vertex_targets, earth_satellite):
        """
        Checks the visibility of the footprint.
        Returns `True` if the footprint is visible on the `candidate_day`, 
        `False` otherwise.
    
        Parameters
        ----------
        candidate_day : astropy.time.Time
            The day to check for footprint visibility.
        
        vertex_targets : list of Target
            The vertices of the footprint boundary represented as Target objects.
        
        earth_satellite : skyfield.sgp4lib.EarthSatellite
            The Skyfield EarthSatellite object representing the satellite in operation.
    
        Returns
        -------
        bool
            True if the footprint is visible, False otherwise.
        """
        for vertex in vertex_targets:
            # Check if the footprint is visible
            # coarse calculation: step through 100 minutes every 2 minutes
            vertex_vis = vertex.get_visibility_windows(
                candidate_day, candidate_day + 100 * u.min, 2 * 60, earth_satellite
            )
            if vertex_vis.num == 0:
                return False
        return True
    
    
    def find_obs_days(self, num_days, VE=params.VE, period=params.PERIOD):
        """
        Determines the day(s) the survey is visible. For daily cadence surveys, 
        `num_days` specifies the number of required observation days.
    
        Parameters
        ----------
        num_days : int
            The number of days for which the user wants to observe the survey. 

        VE : astropy.time.Time
            The Vernal Equinox time of the current year in UTC.
            Example: VE = Time("2024-03-20 03:06:00") 

        period : float
            The satellite's orbital period in minutes.
    
        Returns
        -------
        obs_days : list of astropy.time.Time
            List of observation days when the survey is visible.
        """
        self.num_days = num_days

        print("Finding possible observation days...")
        # Determine the days when the small survey field is visible 
        # (i.e., when the boresight is not pointing too close to the Sun)
        code_start = timeit.default_timer()
        
        # Convert `boundary_coords` to a list of Target objects 
        # representing the survey's rectangular bounds
        minRA, minDEC, maxRA, maxDEC = Polygon(self.boundary_coords).bounds
        vertex_targets = [
            InertialTarget(RA, DEC) for RA in [minRA, maxRA] for DEC in [minDEC, maxDEC]
        ]
        # Define the RA of the survey center
        surveyRA = (minRA + maxRA) / 2
    
        # Find the day when the survey is in the anti-Sun direction as seen from Earth
        anti_SunRA_day = find_anti_SunRA_day(surveyRA, VE)
       
        # Note: for cadence surveys requiring daily observations over months,
        # start the visible day search from `anti_SunRA_day`
        start_day = anti_SunRA_day

        # To search for all the visible days
        # start from the two ends centered on `start_day`
        if num_days > 170:
            buffer = 2  # Add two extra days on both ends
        else:
            buffer = 0
        
        # Adjust `start_day` range to ensure the length matches `num_days`
        half_days = num_days // 2
        if num_days % 2 == 0:
            first_day = start_day - (half_days - 1 + buffer) * u.day
            last_day = start_day + (half_days + buffer) * u.day
        else:
            first_day = start_day - half_days * u.day
            last_day = start_day + half_days * u.day
        
        # Move forward/backward in time until the footprint is visible on 
        # `first_day`/`last_day`
        while not self._footprint_is_visible(
            first_day, vertex_targets, self.satellite.earth_satellite
        ):
            first_day += 1 * u.day
        while not self._footprint_is_visible(
            last_day, vertex_targets, self.satellite.earth_satellite
        ):
            last_day -= 1 * u.day

        # Create the list of visible days
        num_vis_days = int((last_day - first_day).value) + 1
        if num_vis_days < num_days:
            num_days = num_vis_days
        day_range = np.arange(0, num_days, 1)
        obs_days = first_day + day_range * u.day
    
        code_runtime = timeit.default_timer() - code_start
        print("code runtime:", print_time(code_runtime))
        
        return obs_days
    
    
    def tile(
        self, obs_day, min_intersect_ratio=0.5, rotation_angle=None,
        tile_overlap=params.TILE_OVERLAP, l=params.DETECTOR_LENGTH, 
        w=params.DETECTOR_WIDTH, g=params.DETECTOR_ARRAY_GAP
    ):
        """
        Tiles the survey footprint and order the tiles in snake scan sequence, 
        i.e. a zigzag pattern. Each tile represents the coverage of a single 
        telescope pointing. 
        
        The default tile rotation angle corresponds to the spacecraft's optimal 
        roll angle on a particular `obs_day`, where the body-mounted solar arrays 
        are better oriented to face the Sun.

        Note that the telescope's focal plane reference frame, which defines the 
        orientation of the tile in the body frame, is specified as follows: the 
        origin is located at the tile's center, the +X axis points into the page, 
        the +Y axis points to the left, and the +Z axis points upward.
        
        Parameters
        ----------
        obs_day : astropy.time.Time
            The observation day, formatted as Time("YYYY-MM-DD").
        
        min_intersect_ratio : float, default=0.5
            Minimum required intersection ratio between a tile and the footprint 
            to be included. Must be between 0 and 1.
        
        rotation_angle : float, optional
            Tile rotation angle in degrees, clockwise about the optical axis.

        tile_overlap : float, optional
            Overlapping length between adjacent tiles in degrees.
    
        l : float, optional
            Length of each detector in degrees.
        
        w : float, optional
            Width of each detector in degrees.
        
        g : float, optional
            Gap between detectors in degrees.

        Returns
        -------
        `Tiling` instance
        """

        # Grid the survey footprint and order in snake scan sequence
        # note: the grid is shifted to center at the origin
        ra_grid_flat, dec_grid_flat, ra_center_grid, dec_center_grid = snake_scan_grid(
            self.boundary_coords, tile_overlap, l, w, g
        )

        # Compute the optimal tile rotation angle
        if rotation_angle is None:
            # Find the center of the survey field
            minRA, minDEC, maxRA, maxDEC = Polygon(self.boundary_coords).bounds
            centerRA = (minRA + maxRA) / 2
            centerDEC = (minDEC + maxDEC) / 2
            survey_center = InertialTarget(centerRA, centerDEC)

            rotation_angle = optimal_roll_angle(
                survey_center, obs_day, self.satellite.earth_satellite, 
                self.satellite.telescope_boresight, self.satellite.solar_array
            )
            print(
            f"The tiles are rotated by {rotation_angle}° "
            "to better align the body-mounted solar arrays with the Sun."
            )
        
        # Create tiles based on the desired orientation
        # and cover the footprint based on `min_intersect_ratio`
        tile_centers, tile_shapes = orient_tiles(
            self.boundary_coords, rotation_angle, ra_grid_flat, dec_grid_flat, 
            ra_center_grid, dec_center_grid, min_intersect_ratio, l, w, g
        )

        # Convert the list of tile center coordinates to a list of tile target objects 
        self.tile_targets = [InertialTarget(RA, DEC) for RA, DEC in tile_centers]

        for tile, shape in zip(self.tile_targets, tile_shapes):
            # Document each tile's rotation angle and MultiPolygon object as attributes 
            tile.rotation_angle = rotation_angle
            tile.mp_object = shape
        
        return Tiling(self.boundary_coords, self.tile_targets)


    def observe(self, obs_day, min_other_obs_time=60, data_threshold=100, 
                period=params.PERIOD):
        """
        Implements the entire observation process for a single visit of the survey.

        This function handles the scheduling of:
            - Exposures
            - Target occultations
            - Operational tasks, including:
                - spacecraft slewing and settling
                - guide star acquisitions
                - dithers
                - data readouts 
                - data downlinks
        And tracks the onboard battery energy level and data accumulation throughout the
        observation.

        The function calculates and outputs the survey's observing efficiency and
        provides a breakdown of total clock time allocations in a dictionary, including:
            - Time spent on other surveys
            - Idle time
            - Time spent on each operation tasks
        Note: Downlinks are assumed to occur simultaneously with other operational tasks, 
        except for slews.
        
        Parameters
        ----------
        obs_day : astropy.time.Time
            The observation day, formatted as Time("YYYY-MM-DD").

        min_other_obs_time : float, default=60
            The minimum continuous observation time (in seconds) required for the other
            surveys.

        data_threshold : float, default=100
            The onboard data threshold (in GB) required to schedule a data downlink.

        period : float, optional
            The satellite's orbital period in minutes.

        Returns
        -------
        obs_results : dict
            A dictionary that compiles the survey's observation results on the chosen 
            observation day.
            - "obs_efficiency": float
                The observing efficiency (%).
            - "total_clock_time": float
                The total clock time (seconds).
            - "other_survey_time": float
                Time spent on other surveys (seconds).
            - "total_survey_time": float
                Total time spent on this survey, including idle time (seconds).
            - "idle_time": float
                Time spent idle (seconds).
            - "slew_time": float
                Total time spent on slewing (seconds).
            - "settle_time": float
                Total time spent on settling (seconds).
            - "GSA_time": float
                Total time spent acquiring guide stars (seconds).
            - "exposure_time": float
                Total exposure time (seconds).
            - "dither_time": float
                Total time spent dithering (seconds).
            - "readout_time": float
                Total data readout time (seconds).

        battery_tracker : dict
            A dictionary structured as {"time":[], "energy_level":[]} that tracks 
            the onboard battery energy level (in W-hr) at different times throughout 
            the observation. 
            - "time": list of str
                The timestamps when the data onboard is recorded, formatted in 
                ISO 8601: "YYYY-MM-DD HH:MM:SS.sss…".
            - "energy_level": list of floats 
                The onboard battery energy level (in W-hr) at each timestamp.
        
        data_tracker : dict
            A dictionary structured as {"time":[], "data_onboard":[]} that tracks
            the amount of data onboard (in GB) at different times throughout the 
            observation. 
            - "time": list of str
                The timestamps when the data onboard is recorded, formatted in 
                ISO 8601: "YYYY-MM-DD HH:MM:SS.sss…".
            - "data_onboard": list of floats 
                The amount of data onboard (in GB) at each timestamp.
        
        obs_sequence : list
            A list of tile Target objects (each with updated `status` attribute) 
            ordered in the observing sequence.
        """
        # Stores the results of the chosen observation day
        obs_results = {}
        
        if self.tile_targets is None:
            raise TypeError("The survey must be tiled first using the `.tile()` method.")
        
        # Create a copy of the tile targets
        tiles = copy.deepcopy(self.tile_targets)

        # Start observing at the start of the tile's new visible cycle
        try: 
            start_time = tiles[0].get_visibility_windows(
                obs_day, obs_day + period * u.min, 
                60, self.satellite.earth_satellite
            ).get_start_time()
        except ValueError as e:
            # No visibiity window on this day
            raise ValueError(f"The survey is not visible on {obs_day}.")

        Now = start_time
        print(f"Start observing at {Now}")

        # Define ground station target objects
        ground_targets = [EarthFixedTarget(lat, lon) for lat, lon in self.ground_stations]
        print(
            f"Downlinking in {self.downlink_freq}-band at {self.downlink_rate} GBps "
            f"with {len(ground_targets)} ground stations."
        )

        # Define the onboard battery's energy level at the start of the observation
        # assumed to be fully charged
        battery_level = self.battery_cap
           
        # Define Trackers
        Slew = 0 # tracks the total time spent on slewing
        Settle = 0 # tracks the total time spent on settling
        Acquire_Guide_Star = 0 # tracks the total time spent on acquiring guide stars
        Exposure = 0 # tracks the total exposure time
        Dither = 0 # tracks the total time spent on dithering
        Readout = 0 # tracks the total data readout time
        Data_Onboard = 0 # tracks the data accumulated onboard
        # Stores the total time segments (in seconds) 
        # allocated to observing other surveys 
        # (including the time to slew, settle, and acquire a new guide star)
        other_survey_segments = []
        # Stores the idle time segments in seconds
        idle_time_segments = []
        # Tracks when the last time data generated internally is accounted for 
        prev_internal_data_check = start_time
        # Tracks when the last daily downlink windows search is 
        end_search = start_time
            
        # Tracks the onboard battery energy level throughout the observation
        battery_tracker = {"time":[], "energy_level":[]}
        battery_tracker["time"].append(Now.iso)
        battery_tracker["energy_level"].append(battery_level)
        
        # Tracks the amount of data onboard throughout the observation
        data_tracker = {"time":[], "data_onboard":[]}
        data_tracker["time"].append(Now.iso)
        data_tracker["data_onboard"].append(Data_Onboard)
        
        # Stores tiles in the observing sequence 
        obs_sequence = []
        # Index of the tile being imaged
        tile_num = 0 # stored as tile.index 

        # track code runtime
        code_start = timeit.default_timer()
        
        while tiles:
            # print("Time Elapsed:", print_time((Now - start_time).sec))
            
            # Search for the downlink windows every 24 hrs
            if Now >= end_search:
                # print("Searching for downlink windows...")
                search_code_start = timeit.default_timer()
                start_search = Now
                end_search = start_search + 1 * u.day
                downlink_windows = Window()
                
                # At each ground station
                for station in ground_targets: 
                    try:
                        contact_windows = station.get_downlink_windows(
                            start_search, end_search, self.min_elevation, 
                            self.satellite.earth_satellite, self.max_cloud_cover
                        )
                    except ValueError as e:
                        # Insufficient cloud coverage data
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                    
                    # Filter out contact times less than 1 minute
                    contact_windows.filter(1*60)
                    # Add to the total downlink windows in a day
                    downlink_windows += contact_windows

                # Sort the downlink windows in chronological order
                # and merge pass times over different ground stations if they overlap
                downlink_windows.merge()
                runtime = timeit.default_timer() - search_code_start
                # print(
                #     f"There are {downlink_windows.num} downlink windows from "
                #     f"{start_search} to {end_search}"
                # )
                # print("runtime:", print_time(runtime))
    
            # Select the tile in snake-scan order
            tile = tiles.pop(0)
            
            # Define slew
            if tile_num == 0:
                # Initial observation
                # Assume the satellite slews from its CVZ center
                
                # Compute the initial pointing quaternion
                CVZ = InertialTarget(
                    *find_CVZcenter(self.satellite.earth_satellite, Now)
                )
                sun_pointing = SolarSystemTarget('sun').pointing(
                    self.satellite.earth_satellite, Now
                )
                CVZ.q_attitude = get_attitude(
                    np_eci=CVZ.pointing(), np_b=self.satellite.telescope_boresight,
                    ns_eci=sun_pointing, ns_b_goal=self.satellite.solar_array
                )

                # Compute the final pointing quaternion
                tile.q_attitude = tile_pointing_attitude(
                    tile, self.satellite.telescope_boresight
                )

                # Determine the slew time and the power generation at final pointing
                slew = perform_slew(
                    CVZ, tile, self.satellite.earth_satellite, Now, 
                    self.satellite.telescope_boresight,
                    self.solar_panel_normals, self.solar_panel_areas, 
                    self.solar_panel_eff, self.power_load, 
                    self.satellite.slew_angle_data, self.satellite.slew_time_data
                )
                
            else:
                # Subsequent observations
                
                # Compute the initial pointing quaternion
                prev_tile.q_attitude = tile_pointing_attitude(
                    prev_tile, self.satellite.telescope_boresight
                )

                # Compute the final pointing quaternion
                tile.q_attitude = tile_pointing_attitude(
                    tile, self.satellite.telescope_boresight
                )
                
                # Determine the slew time and the power generation at final pointing
                slew = perform_slew(
                    prev_tile, tile, self.satellite.earth_satellite, Now, 
                    self.satellite.telescope_boresight,
                    self.solar_panel_normals, self.solar_panel_areas, 
                    self.solar_panel_eff, self.power_load, 
                    self.satellite.slew_angle_data, self.satellite.slew_time_data
                )
        
            # Assign an index to the tile for future reference
            if tile.index is None:
                tile_num += 1
                tile.index = tile_num
                
            # Calculate the selected tile's occultation if it's undefined or outdated
            if (
                (tile.orbit_vis is None) or 
                (tile.orbit_vis and tile.orbit_vis.get_start_time() < Now)
            ):
                try:
                    tile.occultation = find_occultation(
                        tile, Now, slew + self.settle + self.acquire_guide, 
                        self.subexposure, period, self.satellite.earth_satellite
                    )
                except ValueError as e:
                    # No more visibility windows
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence

            # Check the selected tile's occultation and determine its status
            if tile.occultation > 0:  # If the tile is currently occulted
                # Find the slew time from the previous tile 
                # to other surveys located at the CVZ center
                
                # Compute the initial pointing quaternion
                prev_tile.q_attitude = tile_pointing_attitude(
                    prev_tile, self.satellite.telescope_boresight
                )
                
                # Compute the final pointing quaternion
                CVZ = InertialTarget(
                    *find_CVZcenter(self.satellite.earth_satellite, Now)
                )
                sun_pointing = SolarSystemTarget('sun').pointing(
                    self.satellite.earth_satellite, Now
                )
                CVZ.q_attitude = get_attitude(
                    np_eci=CVZ.pointing(), np_b=self.satellite.telescope_boresight,
                    ns_eci=sun_pointing, ns_b_goal=self.satellite.solar_array
                )

                # Determine the slew time and the power generation at final pointing
                slew_to_CVZ = perform_slew(
                    prev_tile, CVZ, self.satellite.earth_satellite, Now, 
                    self.satellite.telescope_boresight,
                    self.solar_panel_normals, self.solar_panel_areas, 
                    self.solar_panel_eff, self.power_load, 
                    self.satellite.slew_angle_data, self.satellite.slew_time_data
                )
                # print("`slew_to_CVZ`:", print_time(slew_to_CVZ))

                # Find the observation time available for the other survey
                # Note: Assume the slew back time is the same as 'slew_to_CVZ'
                other_obs_time = (
                    tile.occultation - 
                    (slew_to_CVZ + self.settle + self.acquire_guide) * 2
                )

                # If the occultation time is long enough to allow slewing to and 
                # back from the other survey, then allocate the slew to and the 
                # obervation time to the other survey 
                if other_obs_time > min_other_obs_time:
                    tile.status = "other"

                    # Update the battery level based on the net power generated
                    # while pointing at the CVZ
                    # Note: assume the net power generated while slewing to the CVZ
                    # is the same as when pointing at the CVZ
                    duration = (
                        slew_to_CVZ + 
                        self.settle + self.acquire_guide + other_obs_time
                    )
                    try:
                        battery_level = update_battery(
                            battery_level, battery_tracker, Now, duration, 
                            CVZ.net_power, self.battery_cap, self.battery_dod
                        )
                    except ValueError as e:
                        # Battery discharge exceeds the depth-of-discharge limit
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        return obs_results, battery_tracker, data_tracker, obs_sequence
            
                    # Update `Now` to be after slewing to the other survey
                    Now += slew_to_CVZ * u.s

                    # Check for downlink opportunities during the `check_period`
                    # Note: Avoid downlinking while slewing
                    check_period = (
                        self.settle + self.acquire_guide + other_obs_time
                    )
                    try:
                        Data_Onboard = downlinking(
                            check_period, Now, Data_Onboard, data_tracker, 
                            downlink_windows, data_threshold,
                            self.onboard_data_cap, self.downlink_rate, 
                            min_downlink_time=60
                        )
                    except ValueError as e:
                        # Onboard data exceeds storage capacity
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        return obs_results, battery_tracker, data_tracker, obs_sequence

                    # Document the time spent on the other survey
                    other_survey = (
                        slew_to_CVZ + self.settle + 
                        self.acquire_guide + other_obs_time
                    )
                    other_survey_segments.append(other_survey)
                    # print("Other survey time:", print_time(other_survey))
                    
                    # Update `Now` to be after observing the other survey
                    Now += (self.settle + self.acquire_guide + other_obs_time) * u.s
                    # print("Time Elapsed:", print_time((Now - start_time).sec))
            
                    # Define the slew back from other surveys
                    slew = slew_to_CVZ
                    # print("slew back from other surveys")

                # Or else count the occultation time as idle time 
                # (and not slew to other surveys)
                else:
                    tile.status = "idle"
                    idle = tile.occultation
                    # Subtract the time to slew back to the survey from idle time
                    # (if applicable)
                    if slew != 0:
                        idle -= slew + self.settle + self.acquire_guide
                    idle_time_segments.append(idle)
                    # print("Add idle time:", print_time(idle))
            
                    # Update the battery level based on the net power generated
                    # while in the idle state
                    duration = idle
                    try:
                        battery_level = update_battery(
                            battery_level, battery_tracker, Now, duration, 
                            tile.net_power, self.battery_cap, self.battery_dod
                        )
                    except ValueError as e:
                        # Battery discharge exceeds the depth-of-discharge limit
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                        
                    # Check for downlink opportunities during the idle time
                    check_period = idle
                    try:
                        Data_Onboard = downlinking(
                            check_period, Now, Data_Onboard, data_tracker, 
                            downlink_windows, data_threshold,
                            self.onboard_data_cap, self.downlink_rate, 
                            min_downlink_time=60
                        )
                    except ValueError as e:
                        # Onboard data exceeds storage capacity
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        return obs_results, battery_tracker, data_tracker, obs_sequence
            
                    # Update `Now` to be after idle time
                    Now += idle * u.s
                    # print("Time Elapsed:", print_time((Now - start_time).sec))
                            
            else:
                tile.status = "visible"
        
            # Account for slew, settle, and guide star acquisition (if applicable)
            if slew != 0:
                Slew += slew
                Settle += self.settle
                Acquire_Guide_Star += self.acquire_guide

                # Update the battery level based on the net power generated
                # while pointing at the tile
                # Note: assume the net power generated while slewing to the tile
                # is the same as when pointing at the tile
                duration = (
                    slew + 
                    self.settle + self.acquire_guide
                )
                try:
                    battery_level = update_battery(
                        battery_level, battery_tracker, Now, duration, 
                        tile.net_power, self.battery_cap, self.battery_dod
                    )
                except ValueError as e:
                    # Battery discharge exceeds the depth-of-discharge limit
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence
                        
                # Update time
                Now += slew * u.s
            
                # Check for downlink opportunities during the `check_period`
                # Note: Avoid downlinking while slewing
                check_period = self.settle + self.acquire_guide
                try:
                    Data_Onboard = downlinking(
                        check_period, Now, Data_Onboard, data_tracker, 
                        downlink_windows, data_threshold,
                        self.onboard_data_cap, self.downlink_rate, 
                        min_downlink_time=60
                    )
                except ValueError as e:
                    # Onboard data exceeds storage capacity
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence
            
                # Update time
                Now += (self.settle + self.acquire_guide) * u.s
                # print("slew:", print_time(slew))
                # print(f"tile {tile.index} is at ({tile.ra:.3f}°, {tile.dec:.3f}°)")
                # print("settle + guide star acquisition:", 
                #       print_time(self.settle + self.acquire_guide))
                # print("Time Elapsed:", print_time((Now - start_time).sec))

            # Track if the tile is imaged at least once in this cycle
                imaged = False
            
            # Exposure and data readout loop
            while tile.subexposure_count < self.subexposure_num:    
                # Check for occultation
                try:
                    tile.occultation = find_occultation(
                        tile, Now, 0, self.subexposure, period,
                        self.satellite.earth_satellite
                    ) 
                except ValueError as e:
                    # No more visibility windows
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence
                    
                # Stop imaging and reselect tile if occultation occurs
                if tile.occultation > 0:
                    # print(f"tile {tile.index} is currently occulted")
                    tiles.insert(0, tile)
                    break 

                # Track if the tile is imaged at least once in this cycle
                imaged = True  
                    
                # Imaging
                Exposure += self.subexposure
                Dither += self.dither / self.subexposure_num
                Readout += self.readout
                
                # Data generated from image readout 
                # (accounts for data compression if applicable)
                Data_Onboard += self.data / self.data_compression_fac

                # Update the battery level based on the net power generated
                # while pointing at the tile
                duration = (
                    self.subexposure + self.dither / self.subexposure_num +
                    self.readout
                )
                try:
                    battery_level = update_battery(
                        battery_level, battery_tracker, Now, duration, 
                        tile.net_power, self.battery_cap, self.battery_dod
                    )
                except ValueError as e:
                    # Battery discharge exceeds the depth-of-discharge limit
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence

                # Check for downlink opportunities during imaging
                check_period = (
                    self.subexposure + self.dither / self.subexposure_num +
                    self.readout
                )
                try:
                    Data_Onboard = downlinking(
                        check_period, Now, Data_Onboard, data_tracker, 
                        downlink_windows, data_threshold,
                        self.onboard_data_cap, self.downlink_rate, 
                        min_downlink_time=60
                    )
                except ValueError as e:
                    # Onboard data exceeds storage capacity
                    print()
                    print(f"\033[1mError:\033[0m {e}")
                    return obs_results, battery_tracker, data_tracker, obs_sequence
                
                # Update time
                Now += (
                    self.subexposure + self.dither / self.subexposure_num + 
                    self.readout
                ) * u.s
                # print("imaging + dither per subexposure:", 
                #       self.subexposure + self.dither / self.subexposure_num + 
                #       self.readout, "s")
                        
                tile.subexposure_count += 1
            
            # Document the previous tile
            prev_tile = tile

            # Document the observing sequence if the tile was imaged at least once
            if imaged:
                # Append a copy of the tile 
                # so status changes won't affect the copy in the list
                obs_sequence.append(copy.deepcopy(tile))
        
            # Account for internally generated data
            Data_Onboard += (
                (Now - prev_internal_data_check).sec * self.internal_data_rate
            )
            prev_internal_data_check = Now

            # Document the amount of data currently onboard
            data_tracker["time"].append(Now.iso)
            data_tracker["data_onboard"].append(Data_Onboard)

        # Results of this observation day
        # Total clock time in seconds
        Total_Clock_Time = (Now - start_time).sec 
        # Total time spent on other surveys
        Other_Survey_Time = sum(other_survey_segments) 
        # Total idle time in seconds
        Idle_Time = sum(idle_time_segments)  
        # Total survey time (including idle)
        This_Survey_Time = Total_Clock_Time - Other_Survey_Time  
        # Observing efficiency (%)
        obs_efficiency = (Exposure / This_Survey_Time) * 100  

        # Print the results
        print(f"\033[1mObserving efficiency: {obs_efficiency:.2f} %\033[0m")
        print("Total clock time:", print_time(Total_Clock_Time))
        print("Total time spent on other surveys:", print_time(Other_Survey_Time))
        print("Total time spent on this survey:", print_time(This_Survey_Time))
        print(
            f"   Exposure time ({((Exposure/ This_Survey_Time) * 100):.2f} %):", 
            print_time(Exposure)
        )
        print(
            f"   Idle time ({((Idle_Time/ This_Survey_Time) * 100):.2f} %):", 
            print_time(Idle_Time)
        )
        print(
            f"   Slew time ({((Slew/ This_Survey_Time) * 100):.2f} %):", 
            print_time(Slew)
        )
        
        # Code runtime 
        code_runtime = timeit.default_timer() - code_start
        print("code runtime:", print_time(code_runtime))
        print()

        # Stores the results
        obs_results = {
            "obs_efficiency": obs_efficiency,
            "total_clock_time": Total_Clock_Time,
            "other_survey_time": Other_Survey_Time,
            "total_survey_time": This_Survey_Time,
            "idle_time": Idle_Time,
            "slew_time": Slew,
            "settle_time": Settle,
            "GSA_time": Acquire_Guide_Star,
            "exposure_time": Exposure,
            "dither_time": Dither,
            "readout_time": Readout
        }
            
        return obs_results, battery_tracker, data_tracker, obs_sequence


class WideSurvey(Survey):
    """
    A wide survey covers a large, contiguous area, typically around 1000 square degrees
    or more. 
    """
    
    def __init__(
        self,
        satellite,
        boundary_coords, 
        exposure, 
        downlink_freq,
        subexposure_num=params.DITHER_POINT_NUM, 
        settle=params.SETTLE_TIME, 
        acquire_guide=params.GUIDE_STAR_ACQUISITION_TIME,
        dither=params.DITHER_TIME, 
        readout=params.READOUT_TIME, 
        data=params.DATA,
        data_compression_fac=params.DATA_COMPRESSION_FAC,
        internal_data_rate=params.INTERNAL_DATA_RATE,
        onboard_data_cap=params.ONBOARD_DATA_CAP,
        downlink_rate_kwargs=params.DOWNLINK_RATE,
        min_elevation_kwargs=params.MIN_ELEVATION,
        max_cloud_cover_kwargs=params.MAX_CLOUD_COVER,
        ground_stations_kwargs=params.GROUND_STATIONS,
        solar_panel_normals=params.SOLAR_PANEL_NORMALS,
        solar_panel_areas=params.SOLAR_PANEL_AREAS,
        solar_panel_eff=params.SOLAR_PANEL_EFF,
        battery_cap=params.BATTERY_CAP,
        battery_dod=params.BATTERY_DOD,
        power_load=params.POWER_LOAD,
    ):
        """
        Create a WideSurvey instance.
        All parameters are stored as attributes.
        
        Parameters
        ----------
        satellite : Satellite 
            The satellite observing the survey.
        
        boundary_coords: np.ndarray
            An array of survey footprint's boundary coordinates as (ra, dec) in degrees.

        exposure : float
            The exoposure (or open-shutter) time per tile in seconds. 

        downlink_freq : str
            The frequency band selected for data downlink (e.g., "X", "Ka", or "optical").

        subexposure_num : int
            Number of dither points to improve sub-pixel sampling.

        settle : float
            The time, in seconds, for the spacecraft to settle after slewing
            and for the Fine Guidance Sensor (FGS) to scan for targets.

        acquire_guide : float
            The time, in seconds, for the FGS to lock onto guide stars.

        dither : float
            The total time, in seconds, to perform the dither pattern for each tile. 

        readout : float
            Data readout time in seconds.

        data : float
            The amount of data, in GB, obtained per subexposure readout. 
        
        data_compression_fac : int
            The onboard data compression factor.

        internal_data_rate : float
            The internal data generation rate, in GBps, due to operations. 

        onboard_data_cap : float
            The onboard data capacity in GB.

        downlink_rate_kwargs : dict
            Dictionary specifying the data downlink rate (in GBps) for each frequency 
            band.

        min_elevation_kwargs : dict
            Dictionary defining the minimum satellite elevation angle (in degrees) 
            required to establish contact with ground stations for each frequency band.

        max_cloud_cover_kwargs : dict
            Dictionary specifying the maximum cloud coverage (in percentage) permitted 
            for optical downlinking. Set to `None` for non-optical frequency bands.

        ground_stations_kwargs : dict
            Dictionary containing lists of ground station coordinates 
            (latitude, longitude) available for each frequency band.

        solar_panel_normals : list of numpy.ndarray
            A list of unit vectors representing the normal directions of the solar arrays 
            in the spacecraft body frame.
    
        solar_panel_areas : numpy.ndarray
            A 1D array representing the area (in m²) of each solar array, 
            corresponding to `solar_panel_normals`.
    
        solar_panel_eff : float
            End-of-life (EOL) efficiency of the solar panels, expressed as a decimal.
    
        battery_cap : float
            The total energy capacity (in Watt-hours) of the battery. 

        battery_dod : float
             The allowable depth-of-discharge limit for the battery, expressed as a decimal 
             fraction of its total capacity.
    
        power_load : float
            The orbit-average power consumption (in Watts).

        Attributes
        ----------
        downlink_rate : float
            The data downlink rate, in GBps, of the chosen frequency band.

        min_elevation : float
            The minimum satellite elevation, in degrees, to establish contact with the 
            ground stations of the chosen frequency band.

        max_cloud_cover : float or None
            The maximum percentage of cloud coverage that still allows for optical 
            downlinking. `None` for non-optical frequency bands.

        ground_stations : list of tuples
            A list of ground station coordinates (latitude, longitude) available for the
            chosen frequency band.

        tile_targets : list of lists
            Stores the tile center coordinates as IntertialTarget in bins ordered in 
            ascending RA.

        num_days_ahead : int
            The number of days before the first tile's optimal observation day.
            Note: the optimal observation day for a tile is defined as the day when 
            pointing at the tile aligns with the anti-Sun direction. 

        Returns
        -------
        `WideSurvey` instance
        """
        super().__init__()

        # Satellite object
        self.satellite = satellite
        if satellite.earth_satellite is None:
            raise TypeError(
                "The Skyfield EarthSatellite object is not defined. "
                "The `build_from_params()` or `build_from_TLE()` method "
                "must be called on the Satellite instance first."
            )

        # Survey footprint boundary
        self.boundary_coords = boundary_coords
        
        # Imaging specifications
        self.exposure = exposure # exposure time per tile 
        self.subexposure_num = subexposure_num # depends on the dither pattern
       
        # Opertion task times (in seconds)
        self.settle = settle # settle time + FGS(Fine Guidance Sensors) scan ID time
        self.acquire_guide = acquire_guide # guide star acquisition time
        self.dither = dither # an estimate of total (primary + subpx) dither time per tile
        self.readout = readout # read out time per subexposure
        
        # Data 
        self.data = data # amount of data (in GB) obtained per subexposure readout
        self.data_compression_fac = data_compression_fac # onboard data compression factor
        self.internal_data_rate = internal_data_rate # internal generation rate (in GBps)
        self.onboard_data_cap = onboard_data_cap # onboard data capacity (in GB)

        # Downlink parameters
        self.downlink_rate_kwargs = downlink_rate_kwargs
        self.min_elevation_kwargs = min_elevation_kwargs
        self.max_cloud_cover_kwargs = max_cloud_cover_kwargs
        self.ground_stations_kwargs = ground_stations_kwargs
        # Set downlink frequency
        self.downlink_freq = downlink_freq  # Uses the setter to initialize

        # Power
        self.solar_panel_normals = solar_panel_normals # each solar panel normal vectors
        self.solar_panel_areas = solar_panel_areas # each solar panel area (in m²)
        self.solar_panel_eff = solar_panel_eff # end-of-life efficiency 
        self.battery_cap = battery_cap # battery total capacity (in W-hr)
        self.battery_dod = battery_dod # battery depth-of-discharge limit
        self.power_load = power_load # average power consumption (in W)

        # Initialize potential attributes
        self.tile_targets = None # stores list of binned tile Target objects
        self.num_days_ahead = None # records the chosen observation starting day
    
    @property
    def exposure(self):
        """
        Get the current exposure time per tile.
        """
        return self._exposure

    
    @exposure.setter
    def exposure(self, value):
        """
        Set the exposure per tile to `value`.
        """
        self._exposure = value

    
    @property
    def subexposure_num(self):
        """
        Get the number of subexposures per tile 
        (which depends on the dither pattern).
        """
        return self._subexposure_num

    
    @subexposure_num.setter
    def subexposure_num(self, value):
        """
        Set the number of subexposures per tile to `value`. 
        """
        self._subexposure_num = value

    
    @property
    def subexposure(self):
        """
        Calculate the exposure time per subexposure based on
        the set `exposure` and `subexposure_num` attributes.
        """
        return self._exposure / self._subexposure_num

    
    @property
    def downlink_freq(self):
        """
        Get the current downlink frequency band.
        """
        return self._downlink_freq

    
    @downlink_freq.setter
    def downlink_freq(self, new_freq):
        """
        Set the downlink frequency band and update dependent attributes.
        """
        if new_freq not in self.downlink_rate_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the downlink_rate_kwargs dictionary."
            )
            
        if new_freq not in self.min_elevation_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the min_elevation_kwargs dictionary."
            )
        if new_freq not in self.max_cloud_cover_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the max_cloud_cover_kwargs dictionary."
            )
        if new_freq not in self.ground_stations_kwargs:
            raise ValueError(
                f"Key '{new_freq}' is not found "
                f"in the ground_stations_kwargs dictionary."
            )
        
        # Update `downlink_freq`
        self._downlink_freq = new_freq
        
        # Update dependent attributes
        # data downlink rate in GBps
        self.downlink_rate = self.downlink_rate_kwargs[new_freq]
        # satellite minimum elevation to link with ground station
        self.min_elevation = self.min_elevation_kwargs[new_freq]
        # maximum cloud coverage % for optical downlink
        self.max_cloud_cover = self.max_cloud_cover_kwargs[new_freq]
        # list of ground station coordinates 
        self.ground_stations = self.ground_stations_kwargs[new_freq]
        
    
    def __repr__(self):
        return (
            f"<WideSurvey object: exposure={self.exposure} s, "
            f"downlink_freq='{self.downlink_freq}'>"
        )

    
    def tile(
        self, dec_cutoff=None, cut_below=True, bin_size=2, zone_size=2, num_days_ahead=60, 
        num_days_behind=30, min_intersect_ratio=0.5, tile_overlap=params.TILE_OVERLAP, 
        l=params.DETECTOR_LENGTH, w=params.DETECTOR_WIDTH, g=params.DETECTOR_ARRAY_GAP,
        VE=params.VE
    ):
        """
        Tiles the survey footprint, where each tile represents the coverage of a single 
        telescope pointing. The footprint boundary can be modified using the `dec_cutoff`
        parameter if needed. 
        
        The tiles are grouped based on their Right Ascension (RA) coordinates into bins. 
        Within each bin, the tiles are grouped into zones and sorted by Declination (DEC)
        in ascending order. Each zone of tiles is assigned an optimal rotation angle 
        based on the Sun's position on its estimated observation day. The observation day
        can be customized by defining `num_days_ahead` of the first tile and 
        `num_days_behind` the last tile of the survey. 
        
        The tile rotation angle corresponds to the spacecraft's optimal roll angle on the 
        observation day, where the body-mounted solar arrays are better oriented to face 
        the Sun. Note that the telescope's focal plane reference frame, which defines the 
        orientation of the tile in the body frame, is specified as follows: the origin is 
        located at the tile's center, the +X axis points into the page, the +Y axis points
        to the left, and the +Z axis points upward.

        Note: the roll angles are currently only limited to multiples of 90°.
        
        Parameters
        ----------
        dec_cutoff : float, optional
            Trim the footprint at the declination cutoff in degrees.

        cut_below : bool, optional, default=True
            If True, removes the portion of the footprint below the `dec_cutoff`. 
            If False, removes the portion above `dec_cutoff`.

        bin_size : float
            The size of each RA bin in degrees, defining the range of RA values 
            grouped together in a single bin.
    
        zone_size : float
            The size of each DEC zone in degrees, defining the range of DEC values 
            grouped together in a zone within a bin.

        num_days_ahead : int, default=60
            The number of days before the first tile's optimal observation day.
            Note: the optimal observation day for a tile is defined as the day when 
            pointing at the tile aligns with the anti-Sun direction.

        num_days_behind : int, default=30
            The number of days after the the last tile's optimal observation day.
        
        min_intersect_ratio : float, default=0.5
            Minimum required intersection ratio between a tile and the footprint 
            to be included. Must be between 0 and 1.

        tile_overlap : float, optional
            Overlapping length between adjacent tiles in degrees.
    
        l : float, optional
            Length of each detector in degrees.
        
        w : float, optional
            Width of each detector in degrees.
        
        g : float, optional
            Gap between detectors in degrees.

        VE : astropy.time.Time
            The Vernal Equinox time of the current year in UTC.
            Example: VE = Time("2024-03-20 03:06:00")

        Returns
        -------
        `Tiling` instance
        """
        print("Tiling the survey field...")
        code_start = timeit.default_timer()

        # Document the observation starting time
        self.num_days_ahead = num_days_ahead
        
        
        # Modify the footprint if needed
        trimmed_boundary_coords = cut_footprint(
            self.boundary_coords, dec_cutoff, cut_below
        )
        
        # Grid the survey footprint and order in snake scan sequence
        # note: the grid is shifted to center at the origin
        ra_grid_flat, dec_grid_flat, ra_center_grid, dec_center_grid = snake_scan_grid(
            trimmed_boundary_coords, tile_overlap, l, w, g
        )

        # Group the flattened grid (centered at origin) into RA bins
        # where each bin is sectioned into zones based on Dec values
        binned_zones = bin_tiles(
            bin_size, zone_size, ra_grid_flat, dec_grid_flat
        )

        # Determine the observation day at each RA bin
        # Given the observation starts `num_days_ahead` of the first bin's optimal
        # observation time (i.e. anti-sun direction), and ends `num_days_behind`
        # the last bin, assume the offset increases linearly
        num_bins = len(binned_zones)
        offsets = np.linspace(-num_days_ahead, num_days_behind, num_bins)

        binned_tiles = []
        
        for bin, offset in zip(binned_zones, offsets):
            # Store tiles in the same bin
            store_tiles = []

            # For each zone
            for zone in bin:
                # If the zone contains tiles
                if zone:
                    # Unpack RA and DEC
                    zone_ra, zone_dec = zip(*zone)
                    
                    # Determine the true RA and DEC value 
                    # instead of the value about the origin
                    ra_list, dec_list = _rotate_move_grid(
                        zone_ra, zone_dec, 0, ra_center_grid, dec_center_grid
                    )

                    # Determine the zone's center 
                    zone_ra_center = np.mean(ra_list)
                    zone_dec_center = np.mean(dec_list)
                    zone_center =InertialTarget(zone_ra_center, zone_dec_center)

                    # Find optimal roll angle given sun's direction 
                    # when pointing at the zone on the observation day
                    estimated_obs_day = (find_anti_SunRA_day(zone_ra_center, VE) 
                                         + offset * u.day)
                    rotation_angle = optimal_roll_angle(
                        zone_center, estimated_obs_day, self.satellite.earth_satellite, 
                        self.satellite.telescope_boresight, self.satellite.solar_array
                    )

                    # Normalize the optimal `rotation angle` to [-180°, 180°]
                    normalized_angle = (rotation_angle + 180) % 360 - 180
                    # Find the closest rotation that resembles 0° rotation
                    target_angles = [-180, -90, 0, 90, 180]
                    rotation_angle = min(target_angles, key=lambda x: abs(x - normalized_angle))
                    
                    # Orient the the zone's grid of tiles
                    tile_centers, tile_shapes = orient_tiles(
                        trimmed_boundary_coords, rotation_angle, zone_ra, zone_dec, 
                        ra_center_grid, dec_center_grid, min_intersect_ratio, l, w, g
                    )

                    # Order the tile center coordinates in ascending DEC
                    tile_centers = sorted(tile_centers, key=lambda coord: coord[1])
                    # Convert to a list of tile target objects 
                    tiles = [InertialTarget(RA, DEC) for RA, DEC in tile_centers]

                    # Document each tile's rotation angle and MultiPolygon object
                    # in this zone
                    for tile, shape in zip(tiles, tile_shapes):
                        tile.rotation_angle = rotation_angle
                        tile.mp_object = shape

                    # Store the tiles in their bin
                    store_tiles.extend(tiles)

            if store_tiles != []:
                # Collect the bins into one list
                binned_tiles.append(store_tiles)

        # A list of bins
        self.tile_targets = binned_tiles

        code_runtime = timeit.default_timer() - code_start
        print("code runtime:", print_time(code_runtime))

        return Tiling(self.boundary_coords, self.tile_targets)

    
    def observe(
        self, min_other_obs_time=60, data_threshold=100, 
        VE=params.VE, period=params.PERIOD
    ):
        """
        Implements the entire observation process for a single visit of the survey.

        The function handles the scheduling of:
            - Exposures
            - Target occultations
            - Operational tasks, including:
                - spacecraft slewing and settling
                - guide star acquisitions
                - dithers
                - data readouts 
                - data downlinks
        And tracks the onboard battery energy level and data accumulation throughout the 
        observation.

        The observation begins with the bin at the smallest RA value and proceeds 
        sequentially, completing all tiles within each bin before moving to the next.
        For the tiles in each bin, the function determines the most efficient observing 
        sequence to maximize time spent on the survey during the observing season, aiming 
        to observe as many tiles as possible within the available timeframe. 

        The observation start time is dependent on the `num_days_ahead` parameter defined
        when tiling the survey. This parameter determines in how many days before the 
        optimal observation day of the survey's first tile — i.e., the day when pointing 
        at the first tile aligns with the anti-Sun direction — the observation should 
        begin.

        The function calculates and outputs the survey's observing efficiency and 
        provides a breakdown of total clock time allocations in a dictionary, including:
            - Time spent on other surveys
            - Idle time
            - Time spent on each operation tasks
        Note: Downlinks are assumed to occur simultaneously with other operational tasks,
        except for slews.
        
        Parameters
        ----------
        min_other_obs_time : float, default=60
            The minimum continuous observation time (in seconds) required for the other 
            surveys.

        data_threshold : float, default=100
            The onboard data threshold (in GB) required to schedule a data downlink.

        VE : astropy.time.Time, optional
            The Vernal Equinox time of the current year in UTC.
            Example: VE = Time("2024-03-20 03:06:00") 

        period : float, optional
            The satellite's orbital period in minutes.
        
        Returns
        -------
        obs_results : dict of dict
            A nested dictionary that compiles the observation results accumulated after
            the completion of each bin. Each bin is represented as a `str` to serve as the 
            key of the outer dictionary. The value is a dictionary documenting the current 
            observation results with the following keys:
                - "obs_efficiency": float
                    The observing efficiency (%).
                - "total_clock_time": float
                    The total clock time (seconds).
                - "other_survey_time": float
                    Time spent on other surveys (seconds).
                - "total_survey_time": float
                    Total time spent on this survey, including idle time (seconds).
                - "idle_time": float
                    Time spent idle (seconds).
                - "slew_time": float
                    Total time spent on slewing (seconds).
                - "settle_time": float
                    Total time spent on settling (seconds).
                - "GSA_time": float
                    Total time spent acquiring guide stars (seconds).
                - "exposure_time": float
                    Total exposure time (seconds).
                - "dither_time": float
                    Total time spent dithering (seconds).
                - "readout_time": float
                    Total data readout time (seconds).
            The "final" key contains the final observation results after all bins are
            completed.

        battery_tracker : dict
            A dictionary structured as {"time":[], "energy_level":[]} that tracks the
            onboard battery energy level (in W-hr) at different times throughout the
            observation. 
                - "time": list of str
                    The timestamps when the data onboard is recorded, formatted in 
                    ISO 8601: "YYYY-MM-DD HH:MM:SS.sss…".
                - "energy_level": list of floats 
                    The onboard battery energy level (in W-hr) at each timestamp.
            
        data_tracker : dict
            A dictionary structured as {"time":[], "data_onboard":[]} that tracks the
            amount of data onboard (in GB) at different times throughout the observation. 
                - "time": list of str
                    The timestamps when the data onboard is recorded, formatted in 
                    ISO 8601: "YYYY-MM-DD HH:MM:SS.sss…".
                - "data_onboard": list of floats 
                    The amount of data onboard (in GB) at each timestamp.

        obs_sequence : list
            A list of tile Target objects (each with updated `status` attribute) ordered
            in the observing sequence.
        """
        # Stores the observation results after each bin is completed
        # and the final result of the survey
        obs_results = {}
    
        if self.tile_targets is None:
            raise TypeError("The survey must be tiled first using the `.tile()` method.")
    
        # Create a copy of the binned target objects
        bins = copy.deepcopy(self.tile_targets)

        # Define the range of slew times (seconds) allowed for slewing within the survey
        # The lower limit prevents slewing to another tile 
        # with similar viewing efficiency (at similar location)
        min_slew_time = 167 # seconds
        # The upper limit prevents spending too much time slewing
        max_slew_time = 400 # seconds
    
        # Determine the observation starting day
        # the day that is `num_days_ahead` of the day corresponding to when the 
        # smallest RA tile in the survey is in the anti-Sun direction.
        smallest_ra = min(bins[0], key=lambda tile: tile.ra).ra
        start_day = find_anti_SunRA_day(smallest_ra, VE) - self.num_days_ahead * u.day
        print(
            f"Start observing on {start_day.strftime('%Y-%m-%d')}, "
            f"{self.num_days_ahead} days ahead of the first tile."
        )
    
        # Start tiling from the bottom of the first bin
        # note: the tiles are ordered in ascending DEC values in the bins
        top = False
        # Start observing at the start of the tile's new visible cycle
        start_time = bins[0][0].get_visibility_windows(
            start_day, start_day + period * u.min, 
            60, self.satellite.earth_satellite
        ).get_start_time()
        Now = start_time

        # Define ground station target objects
        ground_targets = [EarthFixedTarget(lat, lon) for lat, lon in self.ground_stations]
        print(
            f"Downlinking in {self.downlink_freq}-band at {self.downlink_rate} GBps "
            f"with {len(ground_targets)} ground stations.\n"
        )

        # Define the onboard battery's energy level at the start of the observation
        # assumed to be fully charged
        battery_level = self.battery_cap

        # Define Trackers
        Slew = 0 # tracks the total time spent on slewing
        Settle = 0 # tracks the total time spent on settling
        Acquire_Guide_Star = 0 # tracks the total time spent on acquiring guide stars
        Exposure = 0 # tracks the total exposure time
        Dither = 0 # tracks the total time spent on dithering
        Readout = 0 # tracks the total data readout time
        Data_Onboard = 0 # tracks the data accumulated onboard
        # Stores the total time segments in seconds allocated to observing other surveys 
        # (including the time to slew, settle, and acquire a new guide star)
        other_survey_segments = []
        # Stores the idle time segments in seconds
        idle_time_segments = []
        # Tracks when the last time data generated internally is accounted for 
        prev_internal_data_check = start_time
        # Tracks when the last daily downlink windows search is 
        end_search = start_time

        # Tracks the onboard battery energy level throughout the observation
        battery_tracker = {"time":[], "energy_level":[]}
        battery_tracker["time"].append(Now.iso)
        battery_tracker["energy_level"].append(battery_level)
        
        # Tracks the amount of data onboard throughout the observation
        data_tracker = {"time":[], "data_onboard":[]}
        data_tracker["time"].append(Now.iso)
        data_tracker["data_onboard"].append(Data_Onboard)
        
        # Stores tiles in the observing sequence 
        obs_sequence = []
        # Index of the tile being imaged
        tile_num = 0 # stored as tile.index 
    
        # track code runtime
        code_start = timeit.default_timer()
        prev_bin_ends = start_time

        # Iterate through each bin
        for i, bin in enumerate(bins):
            print(f"Observing bin{i+1}: {len(bin)} tiles")
            bin_code_start = timeit.default_timer()
        
            while bin:
                # print("Time Elapsed:", print_time((Now - start_time).sec))
        
                # Search for the downlink windows every 24 hrs
                if Now >= end_search:
                    # print("Searching for downlink windows...")
                    search_code_start = timeit.default_timer()
                    start_search = Now
                    end_search = start_search + 1 * u.day
                    downlink_windows = Window()
                    
                    # At each ground station
                    for station in ground_targets: 
                        try:
                            contact_windows = station.get_downlink_windows(
                                start_search, end_search, self.min_elevation, 
                                self.satellite.earth_satellite, self.max_cloud_cover
                            )
                        except ValueError as e:
                            # Insufficient cloud coverage data
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence
                            
                        # Filter out contact times less than 1 minute
                        contact_windows.filter(1*60)
                        # Add to the total downlink windows in a day
                        downlink_windows += contact_windows

                    # Sort the downlink windows in chronological order
                    # and merge pass times over different ground stations if they overlap
                    downlink_windows.merge()
                    runtime = timeit.default_timer() - search_code_start
                    # print(
                    #     f"There are {downlink_windows.num} downlink windows from "
                    #     f"{start_search} to {end_search}"
                    # )
                    # print("runtime:", print_time(runtime))
        
                # Select tile and define slew
                if tile_num == 0: # first tile of the survey
                    # Select the initial tile
                    tile = bin.pop(0) if top else bin.pop()

                    # Assume the satellite slews from its CVZ center
                    # Compute the initial pointing quaternion
                    CVZ = InertialTarget(
                        *find_CVZcenter(self.satellite.earth_satellite, Now)
                    )
                    sun_pointing = SolarSystemTarget('sun').pointing(
                        self.satellite.earth_satellite, Now
                    )
                    CVZ.q_attitude = get_attitude(
                        np_eci=CVZ.pointing(), np_b=self.satellite.telescope_boresight,
                        ns_eci=sun_pointing, ns_b_goal=self.satellite.solar_array
                    )
    
                    # Compute the final pointing quaternion
                    tile.q_attitude = tile_pointing_attitude(
                        tile, self.satellite.telescope_boresight
                    )
    
                    # Determine the slew time and the power generation at final pointing
                    slew = perform_slew(
                        CVZ, tile, self.satellite.earth_satellite, Now, 
                        self.satellite.telescope_boresight,
                        self.solar_panel_normals, self.solar_panel_areas, 
                        self.solar_panel_eff, self.power_load, 
                        self.satellite.slew_angle_data, self.satellite.slew_time_data
                    )
                    
                else: # subsequent tiles of the survey
                    # Define the tile candidate in the current tiling direction
                    # Could be a new tile or the previous tile
                    adjacent_tile = bin[0] if top else bin[-1]
                    # At the opposite end of the bin
                    opposite_tile = bin[0] if not top else bin[-1]
                    
                    # Select the tile to observe
                    # Check if the adjacent tile will be visible after slewing to it
                    # Avoid re-calculating previous tile's visibility window
                    if adjacent_tile != prev_tile: 

                        # Determine the slew to `adjacent_tile`
                        # Compute the initial pointing quaternion
                        prev_tile.q_attitude = tile_pointing_attitude(
                            prev_tile, self.satellite.telescope_boresight
                        )
                        # Compute the final pointing quaternion
                        adjacent_tile.q_attitude = tile_pointing_attitude(
                            adjacent_tile, self.satellite.telescope_boresight
                        )
                        # Determine the slew time and the power generation at final pointing
                        adjacent_slew = perform_slew(
                            prev_tile, adjacent_tile, self.satellite.earth_satellite, Now, 
                            self.satellite.telescope_boresight,
                            self.solar_panel_normals, self.solar_panel_areas, 
                            self.solar_panel_eff, self.power_load, 
                            self.satellite.slew_angle_data, self.satellite.slew_time_data
                        )
                        # print("`adjacent_slew`:", print_time(adjacent_slew))

                        # Check adjacent tile's visibility
                        # print("check adjacent tile's current visibility")
                        try:
                            adjacent_tile.occultation = find_occultation(
                                adjacent_tile, Now, 
                                adjacent_slew + self.settle + self.acquire_guide,
                                self.subexposure, period, self.satellite.earth_satellite
                            )
                        except ValueError as e:
                            # No more visibility windows
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence
                        
                        # Select the adjacent tile if it is visible
                        if adjacent_tile.occultation == 0:
                            # print("Select the adjacent tile")
                            tile = bin.pop(0) if top else bin.pop()
                            slew = adjacent_slew
                    
                    # If the current tiling direction is not visible, 
                    # because occultation occurred while imaging the previous tile, 
                    # or because the adjacent tile is not visible
                    if adjacent_tile == prev_tile or adjacent_tile.occultation > 0:
                        
                        # Consider slewing to the opposite tile
                        # Compute the initial pointing quaternion
                        prev_tile.q_attitude = tile_pointing_attitude(
                            prev_tile, self.satellite.telescope_boresight
                        )
                        # Compute the final pointing quaternion
                        opposite_tile.q_attitude = tile_pointing_attitude(
                            opposite_tile, self.satellite.telescope_boresight
                        )
                        # Determine the slew time and the power generation at final pointing
                        opposite_slew = perform_slew(
                            prev_tile, opposite_tile, self.satellite.earth_satellite, Now, 
                            self.satellite.telescope_boresight,
                            self.solar_panel_normals, self.solar_panel_areas, 
                            self.solar_panel_eff, self.power_load, 
                            self.satellite.slew_angle_data, self.satellite.slew_time_data
                        )
                        # print("`opposite_slew`:", print_time(opposite_slew))
                        
                        # Check opposite tile's visibility
                        # print("check opposite tile's current visibility")
                        try:
                            opposite_tile.occultation = find_occultation(
                                opposite_tile, Now, 
                                opposite_slew + self.settle + self.acquire_guide,
                                self.subexposure, period, self.satellite.earth_satellite
                            )
                        except ValueError as e:
                            # No more visibility windows
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence

                        # Select the opposite tile if it is currently visible
                        # or if the slew time is in the desired range
                        if (opposite_tile.occultation == 0 or 
                            min_slew_time < opposite_slew < max_slew_time):
                            # print("Select the tile in the opposite tiling direction")
                            top = not top  # Change tiling direction
                            tile = bin.pop(0) if top else bin.pop()
                            slew = opposite_slew
                        else:
                            # Or else select the tile in the current tiling direction
                            # print("Select the tile in the current tiling direction")
                            tile = bin.pop(0) if top else bin.pop()
                            if tile == prev_tile:
                                slew = 0
                            else:
                                slew = adjacent_slew
        
                # Assign an index to the tile for future reference
                if tile.index is None:
                    tile_num += 1
                    tile.index = tile_num
                
                # Calculate the selected tile's occultation if it's undefined or outdated
                if ((tile.orbit_vis is None) or 
                    (tile.orbit_vis and tile.orbit_vis.get_start_time() < Now)):
                    try:
                        tile.occultation = find_occultation(
                            tile, Now, slew + self.settle + self.acquire_guide, 
                            self.subexposure, period, self.satellite.earth_satellite
                        )
                    except ValueError as e:
                        # No more visibility windows
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                
                # Check the selected tile's occultation and determine its status
                if tile.occultation > 0:  # If the tile is currently occulted
                    # Find the slew time from the previous tile 
                    # to other surveys located at the CVZ center
                    
                    # Compute the initial pointing quaternion
                    prev_tile.q_attitude = tile_pointing_attitude(
                        prev_tile, self.satellite.telescope_boresight
                    )
                    
                    # Compute the final pointing quaternion
                    CVZ = InertialTarget(
                        *find_CVZcenter(self.satellite.earth_satellite, Now)
                    )
                    sun_pointing = SolarSystemTarget('sun').pointing(
                        self.satellite.earth_satellite, Now
                    )
                    CVZ.q_attitude = get_attitude(
                        np_eci=CVZ.pointing(), np_b=self.satellite.telescope_boresight,
                        ns_eci=sun_pointing, ns_b_goal=self.satellite.solar_array
                    )
                    
                    # Determine the slew time and the power generation at final pointing
                    slew_to_CVZ = perform_slew(
                        prev_tile, CVZ, self.satellite.earth_satellite, Now, 
                        self.satellite.telescope_boresight,
                        self.solar_panel_normals, self.solar_panel_areas, 
                        self.solar_panel_eff, self.power_load, 
                        self.satellite.slew_angle_data, self.satellite.slew_time_data
                    )
                    # print("`slew_to_CVZ`:", print_time(slew_to_CVZ))
                
                    # Find the observation time available for the other survey
                    # Note: Assume the slew back time is the same as 'slew_to_CVZ'
                    other_obs_time = (
                        tile.occultation - 
                        (slew_to_CVZ + self.settle + self.acquire_guide) * 2
                    )

                    # If the occultation time is long enough to allow slewing to and 
                    # back from the other survey, then allocate the slew to and the 
                    # obervation time to the other survey 
                    if other_obs_time > min_other_obs_time:
                        tile.status = "other"

                        # Update the battery level based on the net power generated
                        # while pointing at the CVZ
                        # Note: assume the net power generated while slewing to the CVZ
                        # is the same as when pointing at the CVZ
                        duration = (
                            slew_to_CVZ + 
                            self.settle + self.acquire_guide + other_obs_time
                        )
                        try:
                            battery_level = update_battery(
                                battery_level, battery_tracker, Now, duration, 
                                CVZ.net_power, self.battery_cap, self.battery_dod
                            )
                        except ValueError as e:
                            # Battery discharge exceeds the depth-of-discharge limit
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence
                        
                        # Update `Now` to be after slewing to the other survey
                        Now += slew_to_CVZ * u.s

                        # Check for downlink opportunities during the `check_period`
                        # Note: Avoid downlinking while slewing
                        check_period = (
                            self.settle + self.acquire_guide + other_obs_time
                        )
                        try:
                            Data_Onboard = downlinking(
                                check_period, Now, Data_Onboard, data_tracker, 
                                downlink_windows, data_threshold,
                                self.onboard_data_cap, self.downlink_rate, 
                                min_downlink_time=60
                            )
                        except ValueError as e:
                            # Onboard data exceeds storage capacity
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence

                        # Document the time spent on the other survey
                        other_survey = (
                            slew_to_CVZ + self.settle + 
                            self.acquire_guide + other_obs_time
                        )
                        other_survey_segments.append(other_survey)
                        # print("Other survey time:", print_time(other_survey))
                
                        # Update `Now` to be after observing the other survey
                        Now += (self.settle + self.acquire_guide + other_obs_time) * u.s
                        # print("Time Elapsed:", print_time((Now - start_time).sec))
                
                        # Define the slew back from other surveys
                        slew = slew_to_CVZ
                        # print("slew back from other surveys")

                    # Or else count the occultation time as idle time 
                    # (and not slew to other surveys)
                    else:
                        tile.status = "idle"
                        idle = tile.occultation
                        # Subtract the time to slew back to the survey from idle time
                        # (if applicable)
                        if slew != 0:
                            idle -= slew + self.settle + self.acquire_guide
                        idle_time_segments.append(idle)
                        # print("Add idle time:", print_time(idle))

                        # Update the battery level based on the net power generated
                        # while in the idle state
                        duration = idle
                        try:
                            battery_level = update_battery(
                                battery_level, battery_tracker, Now, duration, 
                                tile.net_power, self.battery_cap, self.battery_dod
                            )
                        except ValueError as e:
                            # Battery discharge exceeds the depth-of-discharge limit
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence
                            
                        # Check for downlink opportunities during the idle time
                        check_period = idle
                        try:
                            Data_Onboard = downlinking(
                                check_period, Now, Data_Onboard, data_tracker, 
                                downlink_windows, data_threshold,
                                self.onboard_data_cap, self.downlink_rate, 
                                min_downlink_time=60
                            )
                        except ValueError as e:
                            # Onboard data exceeds storage capacity
                            print()
                            print(f"\033[1mError:\033[0m {e}")
                            # Print the total runtime
                            total_runtime = timeit.default_timer() - code_start
                            print('total runtime:', print_time(total_runtime))
                            return obs_results, battery_tracker, data_tracker, obs_sequence
                
                        # Update `Now` to be after idle time
                        Now += idle * u.s
                        # print("Time Elapsed:", print_time((Now - start_time).sec))
                                
                else:
                    tile.status = "visible"
                
                # Account for slew, settle, and guide star acquisition (if applicable)
                if slew != 0:
                    Slew += slew
                    Settle += self.settle
                    Acquire_Guide_Star += self.acquire_guide

                    # Update the battery level based on the net power generated
                    # while pointing at the tile
                    # Note: assume the net power generated while slewing to the tile
                    # is the same as when pointing at the tile
                    duration = (
                        slew + 
                        self.settle + self.acquire_guide
                    )
                    try:
                        battery_level = update_battery(
                            battery_level, battery_tracker, Now, duration, 
                            tile.net_power, self.battery_cap, self.battery_dod
                        )
                    except ValueError as e:
                        # Battery discharge exceeds the depth-of-discharge limit
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                            
                    # Update time
                    Now += slew * u.s
                
                    # Check for downlink opportunities during the `check_period`
                    # Note: Avoid downlinking while slewing
                    check_period = self.settle + self.acquire_guide
                    try:
                        Data_Onboard = downlinking(
                            check_period, Now, Data_Onboard, data_tracker, 
                            downlink_windows, data_threshold,
                            self.onboard_data_cap, self.downlink_rate, 
                            min_downlink_time=60
                        )
                    except ValueError as e:
                        # Onboard data exceeds storage capacity
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                
                    # Update time
                    Now += (self.settle + self.acquire_guide) * u.s
                    # print("slew:", print_time(slew))
                    # print(f"tile {tile.index} is at ({tile.ra:.3f}°, {tile.dec:.3f}°)")
                    # print("settle + guide star acquisition:", 
                    #       print_time(self.settle + self.acquire_guide))
                    # print("Time Elapsed:", print_time((Now - start_time).sec))

                # Record when the tile begins imaging
                tile.timestamp = Now

                # Track if the tile is imaged at least once in this cycle
                imaged = False
                
                # Exposure and data readout loop
                while tile.subexposure_count < self.subexposure_num:
                    # Check for occultation
                    try:
                        tile.occultation = find_occultation(
                            tile, Now, 0, self.subexposure, period,
                            self.satellite.earth_satellite
                        ) 
                    except ValueError as e:
                        # No more visibility windows
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                        
                    # Stop imaging and reselect tile if occultation occurs
                    if tile.occultation > 0:
                        # print(f"tile {tile.index} is currently occulted")
                        if top:
                            bin.insert(0, tile)
                        else:
                            bin.append(tile)
                        break 

                    # Track if the tile is imaged at least once in this cycle
                    imaged = True  
                
                    # Imaging
                    Exposure += self.subexposure
                    Dither += self.dither / self.subexposure_num
                    Readout += self.readout
                
                    # Data generated from image readout 
                    # (accounts for data compression if applicable)
                    Data_Onboard += self.data / self.data_compression_fac

                    # Update the battery level based on the net power generated
                    # while pointing at the tile
                    duration = (
                        self.subexposure + self.dither / self.subexposure_num +
                        self.readout
                    )
                    try:
                        battery_level = update_battery(
                            battery_level, battery_tracker, Now, duration, 
                            tile.net_power, self.battery_cap, self.battery_dod
                        )
                    except ValueError as e:
                        # Battery discharge exceeds the depth-of-discharge limit
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                    
                    # Check for downlink opportunities during imaging
                    check_period = (
                        self.subexposure + self.dither / self.subexposure_num +
                        self.readout
                    )
                    try:
                        Data_Onboard = downlinking(
                            check_period, Now, Data_Onboard, data_tracker, 
                            downlink_windows, data_threshold,
                            self.onboard_data_cap, self.downlink_rate, 
                            min_downlink_time=60
                        )
                    except ValueError as e:
                        # Onboard data exceeds storage capacity
                        print()
                        print(f"\033[1mError:\033[0m {e}")
                        # Print the total runtime
                        total_runtime = timeit.default_timer() - code_start
                        print('total runtime:', print_time(total_runtime))
                        return obs_results, battery_tracker, data_tracker, obs_sequence
                
                    # Update time
                    Now += (
                        self.subexposure + self.dither / self.subexposure_num + 
                        self.readout
                    ) * u.s
                    # print("imaging + dither per subexposure:", 
                    #       self.subexposure + self.dither / self.subexposure_num + 
                    #       self.readout, "s")
                    
                    tile.subexposure_count += 1
                
                # Document the previous tile
                prev_tile = tile
                
                # Document the observing sequence if the tile was imaged at least once
                if imaged:
                    # Append a copy of the tile 
                    # so status changes won't affect the copy in the list
                    obs_sequence.append(copy.deepcopy(tile))
                
                # Account for internally generated data
                Data_Onboard += (
                    (Now - prev_internal_data_check).sec * self.internal_data_rate
                )
                prev_internal_data_check = Now
        
                # Document the amount of data currently onboard
                data_tracker["time"].append(Now.iso)
                data_tracker["data_onboard"].append(Data_Onboard)
                
            # After a bin is completed, switch the tiling direction
            top = not top
            
            # Update results after each bin is completed
            # Time spent on this bin
            bin_clock_time = (Now - prev_bin_ends).sec  
            # Total clock time in seconds
            Total_Clock_Time = (Now - start_time).sec  
            # Total time spent on other surveys
            Other_Survey_Time = sum(other_survey_segments) 
            # Total idle time in seconds
            Idle_Time = sum(idle_time_segments)  
            # Total survey time (including idle)
            This_Survey_Time = Total_Clock_Time - Other_Survey_Time  
            # Observing efficiency (%)
            obs_efficiency = (Exposure / This_Survey_Time) * 100  
            
            # Print the current results
            print(f"\033[1mbin{i+1} completed\033[0m")
            print(f"Currently observing around RA = {round(tile.ra, 2)} deg")
            print("Time spent completing this bin:", print_time(bin_clock_time))
            print(
                f"\033[1mCumulative observing efficiency: {obs_efficiency:.2f} %\033[0m"
            )
            print("current time:", Now)
            print("current anti-Sun RA day:", find_anti_SunRA_day(tile.ra, VE))
            print("Total clock time:", print_time(Total_Clock_Time))
            prev_bin_ends = Now
            bin_runtime = timeit.default_timer() - bin_code_start
            print("code runtime:", print_time(bin_runtime))
            print()
            
            # Store the results for the current bin
            obs_results[f"bin{i + 1}"] = {
                "obs_efficiency": obs_efficiency,
                "total_clock_time": Total_Clock_Time,
                "other_survey_time": Other_Survey_Time,
                "total_survey_time": This_Survey_Time,
                "idle_time": Idle_Time,
                "slew_time": Slew,
                "settle_time": Settle,
                "GSA_time": Acquire_Guide_Star,
                "exposure_time": Exposure,
                "dither_time": Dither,
                "readout_time": Readout
            }
        
        # Store the final results after all bins are completed
        obs_results["final"] = {
            "obs_efficiency": obs_efficiency,
            "total_clock_time": Total_Clock_Time,
            "other_survey_time": Other_Survey_Time,
            "total_survey_time": This_Survey_Time,
            "idle_time": Idle_Time,
            "slew_time": Slew,
            "settle_time": Settle,
            "GSA_time": Acquire_Guide_Star,
            "exposure_time": Exposure,
            "dither_time": Dither,
            "readout_time": Readout
        }
        
        # Print the total runtime
        total_runtime = timeit.default_timer() - code_start
        print('code total runtime:', print_time(total_runtime))

        return obs_results, battery_tracker, data_tracker, obs_sequence
        