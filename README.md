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

## Run one notebook
To run the analysis for one notebook, you can run
```bash
python run_toi.py <toi id number>
```
where an example `<toi id number> = 724`

To run all the notebooks, you can run
```bash
python run_tois.py
```

## Run tests on template.ipynb
```bash
pytest tests/
```