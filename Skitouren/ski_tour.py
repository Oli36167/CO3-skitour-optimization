import pandas as pd


def read(filename):
    """
    Read csv file with coordinates in LV03
    """
    return pd.read_csv(filename, sep=';')

"""
Times in HOURS
"""
# 1. Schollberg
coordinates_1 = read("schollberg.csv")
time_1 = 3.5

print(coordinates_1)

# 2. Rätschenhorn
coordinates_2 = read("raetschenhorn.csv")
time_2 = 4

# 3. Sulzfluh Überschreitung
coordinates_aufstieg_3 = read("sulzfluh_aufstieg.csv")
coordinates_abfahrt_3 = read("sulzfluh_abfahrt.csv")
time_3 = 5

# 4. Sandhubel
coordinates_4 = read("sandhubel.csv")
time_4 = 4

# 5. Hoch Ducan
coordinates_5 = read("hoch_ducan.csv")
time_5 = 4.5