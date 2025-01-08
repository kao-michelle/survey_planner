## Survey Planning and Scheduling Tool for Space Telescope Missions
This python package is under active development.

This planning and scheduling tool is built for the preparation of the CASTOR mission as it enters Phase A. 
The tool evaluates the feasibility of a space telescope's survey requirements and provides recommendations for alternative observation strategies. 
Given the satellite's orbital parameters and survey specifications, the tool simulates the entire observation process for a single survey visit. 
It schedules exposure (open-shutter) times while accounting for target visibility, and schedules operational tasks including spacecraft slewing and settling, 
guide star acquisition, dithering, and data readouts. Additionally, the tool monitors onboard data accumulation and schedules data downlinks accordingly. 
It also tracks solar array power generation at each spacecraft attitude and monitors the onboard battery level throughout the observation. 
Finally, it outputs the survey's observing efficiency and provides a breakdown of time allocation across all tasks.

### Features
- Includes a built-in Earth satellite orbit propagator for orbital simulations of the telescope.
- Supports four target types:
  - Inertially-fixed targets
  - Solar system objects
  - Earth-orbiting targets
  - Earth-fixed targets
- Determines the optimal tiling strategy for surveying a continuous area of the sky.
- Schedules tasks to maximize survey observing efficiency.
- Incorporates key subsystems:
    - Attitude determination and control
    - Communication (data downlinks)
    - Power generation and battery monitoring

### Dependencies
This project requires the following Python packages:
- `astropy==7.0.0`
- `ipython==8.12.3`
- `matplotlib==3.10.0`
- `numpy==2.2.1`
- `pyquaternion==0.9.9`
- `Requests==2.32.3`
- `sgp4==2.23`
- `Shapely==2.0.6`
- `skyfield==1.49`

You can also view these in the `requirements.txt` file.
