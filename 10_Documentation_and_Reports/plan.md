I need you to download the CORRECT ocean-current dataset for my velocity-compensation algorithm.

DO NOT download another monthly-only dataset as the primary dataset.

## PROJECT REQUIREMENT

My supervisor's requirement is:

"Input is relative velocity.
Output is compensated velocity.
Compensate for the ocean current velocity at the particular depth."

Therefore, the environmental current data must allow us to obtain the ocean-current vector at:

    TIME + LATITUDE + LONGITUDE + DEPTH

The eventual algorithm will be conceptually:

    Relative velocity
             +
    Ocean-current velocity
    at matching time/location/depth
             =
    Compensated velocity

The exact plus/minus sign must NOT be assumed until the relative-velocity coordinate/sign convention is verified.

## DATA I NEED

Download the Copernicus Marine GLOBAL OCEAN PHYSICS current dataset with:

    uo = eastward sea-water velocity
    vo = northward sea-water velocity

The dataset MUST contain:

    time
    latitude
    longitude
    depth
    uo
    vo

Prefer the 6-hourly current product:

    Currents, 6-hourly
    cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i

This is the PRIMARY dataset for the velocity-compensation prototype.

DO NOT substitute it with:
- surface currents only
- monthly currents
- sea-level data
- salinity
- vertical currents
- temperature-only data

## TEMPERATURE

Temperature is also required by my supervisor as an environmental variable.

If practical, also download the matching 6-hourly 3D temperature dataset:

    Temperature, 6-hourly
    cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i

Variable:

    thetao

It should contain:

    time
    latitude
    longitude
    depth
    thetao

IMPORTANT:
Temperature is NOT automatically part of the velocity-compensation equation.
Keep it as an additional environmental variable unless the project specification later requires temperature-dependent correction.

## GEOGRAPHIC REGION

I am studying the Indian Ocean.

Use this broad bounding box unless Copernicus requires a slightly different valid extent:

    West longitude: 20 E
    East longitude: 147 E
    South latitude: 60 S
    North latitude: 30 N

Before downloading, inspect the available product coverage and make sure the selected subset actually covers the Indian Ocean region.

Do NOT download the entire global ocean if Copernicus allows spatial subsetting.

## TIME PERIOD

Do NOT blindly download years of 6-hourly data.

First inspect the available temporal coverage.

For the initial velocity-compensation prototype, select ONE COMPLETE RECENT YEAR that is fully available in the selected product.

Prefer:

    January 1 -> December 31

of the most recent complete year available.

Report the exact start and end dates before downloading.

If the current product does not provide a complete recent year, choose the most recent complete 12-month period available.

## DEPTH

The data must retain the full available vertical dimension.

DO NOT download surface-only current data.

I need the current at different underwater depths because the compensation algorithm will use:

    current(lat, lon, depth, time)

Do not collapse the depth dimension.

After download, report all available depth levels.

## REQUIRED OUTPUT

Save the downloaded/subsetted datasets in a clearly named project directory.

For example:

    Indian_Ocean_Current_Data/
        currents_6hourly.nc
        temperature_6hourly.nc

If the downloaded files have different names, keep the original filenames and document them.

## VERIFY THE DATA AFTER DOWNLOAD

Do NOT simply say "download complete."

Open/inspect the downloaded NetCDF files and report:

1. Filename
2. Dataset size
3. Variables
4. Dimensions
5. Latitude range
6. Longitude range
7. Depth range
8. Number of depth levels
9. Time range
10. Time interval/resolution
11. Units of uo
12. Units of vo
13. Units of thetao
14. Missing/NoData information
15. CRS/coordinate information

Explicitly verify that:

    uo exists
    vo exists
    depth exists
    latitude exists
    longitude exists
    time exists

## DERIVED CURRENT QUANTITIES

Do NOT alter the raw uo/vo data.

We will later calculate:

Current horizontal speed:

    speed = sqrt(uo^2 + vo^2)

Current direction:

    direction = atan2(vo, uo)

Keep uo and vo because the compensation algorithm requires VECTOR velocity, not only scalar speed.

## VERY IMPORTANT

The final algorithm will eventually receive something like:

    timestamp
    vehicle_latitude
    vehicle_longitude
    vehicle_depth
    relative_velocity_east
    relative_velocity_north

The current-data system must then be capable of finding/interpolating:

    uo(time, latitude, longitude, depth)
    vo(time, latitude, longitude, depth)

at the vehicle's corresponding location, depth, and time.

Therefore, preserve the original:

    TIME × DEPTH × LATITUDE × LONGITUDE

dimensions.

## INTERPOLATION

Do NOT implement the compensation algorithm yet.

First download and verify the correct dataset.

However, confirm that the downloaded dataset is suitable for later interpolation in:

    time
    latitude
    longitude
    depth

We will later use interpolation to obtain the current at the exact vehicle position/depth/time when those values do not fall exactly on the Copernicus grid.

## EXISTING DATA

I already downloaded monthly Copernicus data.

DO NOT delete it.

Keep the monthly dataset because it may be useful for:
- seasonal maps
- monthly visualization
- comparison
- background analysis

The new 6-hourly current dataset is the higher-temporal-resolution dataset intended for the velocity-compensation prototype.

## DO NOT MAKE THESE MISTAKES

1. Do NOT download surface currents only.
2. Do NOT download monthly currents as the primary compensation dataset.
3. Do NOT discard depth.
4. Do NOT discard time.
5. Do NOT reduce uo/vo to speed only.
6. Do NOT download global data if spatial subsetting is available.
7. Do NOT claim that temperature is required mathematically for velocity compensation unless justified.
8. Do NOT assume the sign of the compensation equation.
9. Do NOT modify or overwrite my existing monthly dataset.
10. Do NOT start generating fancy maps instead of downloading and verifying the actual data.

## FINAL RESPONSE

After completing the download, report:

    PRIMARY CURRENT DATASET:
    Product:
    Dataset:
    Variables:
    Time resolution:
    Time range:
    Depth range:
    Latitude range:
    Longitude range:
    File location:
    File size:

    TEMPERATURE DATASET:
    Product:
    Dataset:
    Variable:
    Time resolution:
    Time range:
    Depth range:
    Latitude range:
    Longitude range:
    File location:
    File size:

Then explicitly answer:

"Can this dataset provide ocean-current velocity at a particular latitude, longitude, depth, and time for use in a velocity-compensation algorithm?"

If anything prevents this requirement from being satisfied, STOP and explain the problem instead of downloading a different dataset without asking.
