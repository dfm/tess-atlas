#!/usr/bin/env python

# Inspired by:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import codecs
import os
import re

from setuptools import find_packages, setup

# PROJECT SPECIFIC

NAME = "tess_atlas"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "tess_atlas", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = [
    "exoplanet>=0.5.1",
    "pymc3>=3.9",
    "pymc3-ext>=0.1.0",
    "theano-pymc>=1.1.2",
    "celerite2>=0.2.0",
    "lightkurve>=2.0.11",
    "plotly>=4.9.0",
    "arviz>=0.10.0",
    "corner>=2.2.1",
    "seaborn",
    "itables",
    "StrEnum",
]
EXTRA_REQUIRE = {"dev": [
    "pytest>=3.6",
    "testbook>=0.2.3",
    "pre-commit",
    "flake8",
    "black<=21.9b0",
    "isort",
    "jupyter_contrib_nbextensions",
    "sphinx-book-theme>=1.0.0",
    "sphinx_design",
    "sphinx-remove-toctrees",
    "sphinx_togglebutton",
    "jupytext",
    "ipython-autotime",
    "sphinx_external_toc",
    "memory_profiler",
    "pympler",
    "psutil",
    "ploomber-engine>=0.0.24",
    "pretty-jupyter",
    "itables",
    "interruptingcow",
    "jupyter-book",
    'myst-nb'
]}

# END PROJECT SPECIFIC


HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        use_scm_version={
            "write_to": os.path.join(
                "src", NAME, "{0}_version.py".format(NAME)
            ),
            "write_to_template": '__version__ = "{version}"\n',
        },
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_data={
            "tess_atlas": [
                "notebook_controllers/templates/*.py",
                "data/*.csv",
                "webbuilder/template/",
                "slurm_job_generator/templates/*.sh",
                "citations/*.txt",
            ]
        },
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        zip_safe=True,
        entry_points={
            "console_scripts": [
                "run_toi=tess_atlas.notebook_controllers.cli:cli_run_toi",
                "make_webpages=tess_atlas.webbuilder.web_cli:main",
                "make_slurm_job=tess_atlas.slurm_job_generator.slurm_job_generator:main",
                "download_toi=tess_atlas.api.download_analysed_toi:main",
                "update_tic_cache=tess_atlas.data.exofop.update_cache_cli:main",
            ]
        },
    )
