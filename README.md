# TESS-Atlas

<p align="center">
  <img width = "450" src="https://raw.githubusercontent.com/tess-atlas/tess_atlas_webbuilder/main/source/_static/atlas_logo.png" />
  <br>
  <b>TESS Atlas</b>
</p>

The python package used to run the analyses in the TESS-Atlas catalog.

## Installation instructions
To install the package:
```bash
python -m pip install tess-atlas
```
or
```bash
git clone git@github.com:tess-atlas/tess-atlas.git
cd tess-atlas
python -m pip install -e .
```

## How to use

### Analyse a TOI
To run the analysis for one TOI, you can run
```bash
run_toi <toi id number>
```
where an example `<toi id number> = 724`

To only setup the notebook + data needed for the analysis for one TOI, you can run
```bash
run_toi <toi id number> --setup
```


### Downloading results
You can __download completed analyses with
```bash
download_toi 103 --outdir analysed_tois
```

## Publishing `tess_atlas` to pypi
To publish to pypi, you will need admin access to this repo.
Then, publishing just requires you to change the version number `tag` a commit.
The `pypi_release` Github action will (hopefully) take care of the rest.
