[![Unit 🧪 `template.ipynb`](https://github.com/dfm/tess-atlas/workflows/Unit%20%F0%9F%A7%AA%20%60template.ipynb%60/badge.svg)](https://github.com/dfm/tess-atlas/actions?query=workflow%3A%22Unit+%F0%9F%A7%AA+%60template.ipynb%60%22)
[![end-to-end 🧪 `template.ipynb`](https://github.com/dfm/tess-atlas/workflows/end-to-end%20%F0%9F%A7%AA%20%60template.ipynb%60/badge.svg)](https://github.com/dfm/tess-atlas/actions?query=workflow%3A%22end-to-end+%F0%9F%A7%AA+%60template.ipynb%60%22)
[![](https://img.shields.io/badge/Paper-Download-orange)](https://nightly.link/dfm/tess-atlas/workflows/build_paper/paper/main.pdf.zip)

<p align="center">
  <img width = "450" src="docs/_static/atlas_logo.png" />
  <br>
  <b>TESS Atlas</b>
</p>

## Installation instructions
To install the necessary packages, run
```bash
python -m pip install -e .
```

## Instructions to update TESS Atlas
1. Create and analyse some TOIs with the following: `make run`
2. Commit the completed TOIs to a branch and make a PR to main
3. Once PR to main completed, run `make website` to convert the notebooks to HTML and upload to the gh-pages branch and deploy to github-pages

## How to use
To run the analysis for one TOI, you can run
```bash
run_toi <toi id number>
```
where an example `<toi id number> = 724`

To only setup the notebook + data needed for the analysis for one TOI, you can run
```bash
run_toi <toi id number> --setup
```
where an example `<toi id number> = 724`

To run all the notebooks (in batches of 8), you can run
```bash
run_tois
```

To make the slurm files needed to analyse a CSV of TOIs you can run:
```bash
make_slurm_job --toi_csv toi_ids.csv --module_loads "git/2.18.0 gcc/9.2.0 openmpi/4.0.2 python/3.8.5"
```

## Run tests on template.ipynb
```bash
python -m pip install -e ".[test]"
pytest tests/
```
