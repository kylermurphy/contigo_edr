"""Constants for the contigo module

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

# JPL planetary and lunar ephemeris DE440
# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440_tech-comments.txt
jpl_de440 = {'MERCURY':22031.868551e9,
             'VENUS':324858.592000e9,
             'EARTH':398600.435507e9,
             'MARS':42828.375816e9,
             'JUPITER':126712764.100000e9,
             'SATURN':37940584.841800e9,
             'URANUS':5794556.400000e9,
             'NEPTUNE':6836527.100580e9,
             'PLUTO':975.500000e9,
             'MOON':4902.800118e9,
             'SUN':132712440041.279419e9,
             'unit':'M**3/SEC**2'
             }

GMc = {'de440':jpl_de440}

WGS84_EARTH_ANGULAR_VELOCITY = 7.292115e-05
