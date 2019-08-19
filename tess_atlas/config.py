# -*- coding: utf-8 -*-

import os

OUTPUT_DIR = os.environ.get(
    "TESS_ATLAS_OUTPUT_DIR", os.path.join(os.path.abspath("."), "output")
)
