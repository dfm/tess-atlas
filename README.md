<p align="center">
  <img width = "450" src="docs/static/atlas_logo.png" />
  <br>
  <b>TESS Atlas</b>
</p>

## Installation instructions
To intall the necessary packages, run
```bash
pip install -r requirements.txt
```

## Instructions to update TESS Atlas
1. Create and analyse some TOIs with the following: `make run`
2. Commit the completed TOIs to a branch and make a PR to main
3. Once PR to main completed, run `make website` to convert the notebooks to HTML and upload to the gh-pages branch and deploy to github-pages

## Run one notebook
To run the analysis for one notebook, you can run
```bash
python tess_atlas/run_toi.py <toi id number>
```
where an example `<toi id number> = 724`

To run all the notebooks, you can run
```bash
python tess_atlas/run_tois.py
```

## Run tests on template.ipynb
```bash
pytest tests/
```
