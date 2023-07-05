import os

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

DIR = os.path.dirname(__file__)
TIC_CACHE = os.path.join(DIR, "cached_tic_database.csv")
TIC_OLD_CACHE = TIC_CACHE + ".old"

LK_AVAIL = "Lightcurve Availible"
TIC_ID = "TIC ID"
TOI = "TOI"  # 101.01
TOI_INT = "TOI int"  # 101
PLANET_COUNT = "planet count"
MULTIPLANET = "Multiplanet System"
SINGLE_TRANSIT = "Single Transit"
NORMAL = "Normal"
PERIOD = "Period (days)"
