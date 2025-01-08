from astropy.time import Time
import numpy as np

# -------------------------------- SATELLITE PARAMETERS -------------------------------- #
# SPACECRAFT PARAMETERS

# Telescope boresight body-frame direction vector 
TELESCOPE_BORESIGHT = np.array([1, 0, 0]) 

# Antenna body-frame direction vectors
ANTENNAS = [
    np.array([0, np.sin(np.pi / 6), np.cos(np.pi / 6)]),
    np.array([0, np.sin(-np.pi / 6), np.cos(-np.pi / 6)]),
]

# Average solar array normal direction vector in body-frame
# To constraint the roll angle, the average direction of the 
# two body-mounted array normal (-Z) is chosen. 
SOLAR_ARRAY = np.array([0, 0, -1])

# Spacecraft inertia tensor (kg·m²)
SC_INERTIA = np.diag([1000, 2000, 2000]) # kg·m²

# Reaction wheel's maximum angular momentum (N·m·s)
MAX_WHEEL_MOMENTUM = 18 # N·m·s

# Reaction wheel's maximum torque (N·m)
MAX_WHEEL_TORQUE = 0.1  # N·m

# The safety buffer applied to the reaction wheel's performance limits
WHEEL_DESIGN_MARGIN = 4 / 3 


# ORBITAL PARAMETERS

# Current year's Vernal Equinox
VE = Time("2024-3-20 3:6:0") 

# Epoch time
T0 = VE

# Orbital eccentricity
ECCO = 0

# Argument of perigee (in degrees)
ARGPO = 0  # degrees

# Orbital inclination (in degrees)
INCLO = 97.8  # degrees

# Right ascension of ascending node (in degrees)
RAAN = 270  # degrees

# Orbial period (in minutes)
PERIOD = 100.9  # minutes

# Mean anomaly (in degrees)
MO = 0  # degrees

# ----------------------------------- TILE PARAMETERS ---------------------------------- #
# Each tile consists of four detectors arranged in a 2x2 focal-plane array (FPA).

# Adjacent tiles overlapping length (in degrees)
TILE_OVERLAP = 30 / 3600  # degrees

# Each detector's length (in degrees)
DETECTOR_LENGTH = 900 / 3600  # degrees

# Each detector's width (in degrees)
DETECTOR_WIDTH = 860 / 3600  # degrees

# Gap between the detector array (in degrees)
DETECTOR_ARRAY_GAP = 26 / 3600  # degrees

# ------------------------------- OPERATION PARAMETERS -------------------------------- #
# Number of secondary dither pointings depending on the chosen dither pattern
# equivalent to the number of subexposures
DITHER_POINT_NUM = 4

# Time for the spacecraft to settle + FGS(Fine Guidance Sensors) scan ID (in seconds)
SETTLE_TIME = 90  # seconds

# Guide star acquisition time (in seconds)
GUIDE_STAR_ACQUISITION_TIME = 30  # seconds

# Time to perform the dither pattern for each tile
DITHER_TIME = 30  # seconds

# Data readout time (in seconds)
READOUT_TIME = 0.02  # seconds

# Amount of pixels per subexposure taken
PX_AMOUNT = 8600 * 9000 * 4 * 3 

# Bytes per pixel
BYTE_PER_PX = 2

# Amount of data obtained per subexposure readout (in GB)
DATA = (PX_AMOUNT * BYTE_PER_PX) / 10**9  # GB

# Onboard data compression factor for loss-less transmission
DATA_COMPRESSION_FAC = 1

# The internal data generation rate due to operations (in GBps)
INTERNAL_DATA_RATE = 0.0001 / 8  # GBps

# Onboard data capacity (in GB)
ONBOARD_DATA_CAP = 725  # GB

# --------------------------------- DOWNLINK PARAMETERS -------------------------------- #
# Downlink frequency bands: 'X', 'Ka', 'optical'

# Data downlink rate in each frequency band (in GBps)
DOWNLINK_RATE = {
    'X': 1.2 / 8,  # 1.2 Gbps or 2.4 Gbps
    'Ka': 2.4 / 8,  # 2.4 Gbps
    'optical': 10 / 8,  # 10 Gbps
}

# Minimum satellite elevation to establish contact with the ground station (in degrees)
MIN_ELEVATION = {
    'X': 5,  # deg
    'Ka': 5,  # deg
    'optical': 30,  # deg
}

# Downlink weather conditions (only applicable to optical downlinks)
# Set to `None` for non-optical frequency bands
MAX_CLOUD_COVER = {
    'X': None,
    'Ka': None,
    'optical': 30,  # %
}

# The list of ground station coordinates in (latitude, longitude)
# Natural Resource Canada (NRCan) satellite ground stations receiving (S-/X-/Ka-band)
    # GSS = Gatineau Satellite Station (45.5847222, -75.8083333)
    # PASS = Prince Albert Satellite Station (53.2125, -105.9291667)
    # ISSF = Inuvik Satellite Station Facility (68.326375, -133.5415889)
GROUND_STATIONS = {
    'X': [
        (45.5847222, -75.8083333),
        (53.2125, -105.9291667),
        (68.31944444, -133.5488889),
    ],
    'Ka': [
        (45.58583333, -75.80926389),
        (68.326375, -133.5415889),
    ],
    # Mock optical ground station sites
    'optical': [
        (34.1478, -118.1445),
        (37.8213, 22.661),
        (35.6764, 139.65),
        (45.5037, -73.4292),
        (68.3194, -133.5492),
        (48.5204, -123.4188),
    ],
}

# ---------------------------------- POWER PARAMETERS ---------------------------------- #
# There are 4 solar arrays for CASTOR
    # 2 arrays deployed to 90 deg, facing -X (anti-boresight)
    # 1 array body-mounted on hex face (+30 deg to -Z)   
    # 1 array body-mounted on adjacent hex face (-30 deg to -Z)

# Solar array normal direction vectors in body-frame
SOLAR_PANEL_NORMALS = [
    np.array([-1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, -1/2, -np.sqrt(3)/2]),
    np.array([0, 1/2, -np.sqrt(3)/2]),
]

# Area (in meters-squared) of each solar panels
# note: listed in the same order as SOLAR_PANEL_NORMALS
SOLAR_PANEL_AREAS = np.array([1.1, 1.1, 1.1, 1.1]) # m²

# End-of-life (EOL) efficiency of the solar panels, expressed as a decimal
# e.g. for CASTOR, the cell efficiency at BOL is 30.7%, and 27.5% after 7 years.
SOLAR_PANEL_EFF = 0.275

# Onboard battery capacity
# i.e. the available battery capacity (in Watt-hours) to sustain operations.
# e.g. for CASTOR, the battery can supply power to subsystem that is 
# consuming 800 W continuously or up to an hour before it is completely drained.
BATTERY_CAP = 800 # W-hr

# Onboard battery depth-of-discharge limit
# expressed as the fraction of its total capacity
BATTERY_DOD = 0.25

# Orbit-average power consumption
POWER_LOAD = 650 # W

# -------------------------------------------------------------------------------------- #

