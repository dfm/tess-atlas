import os

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

DIR = os.path.dirname(__file__)
TIC_CACHE = os.path.join(DIR, "cached_tic_database.csv")
TIC_OLD_CACHE = TIC_CACHE + ".old"
