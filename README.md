[![](https://img.shields.io/badge/Paper-Download-orange)](https://nightly.link/dfm/tess-atlas/workflows/build_paper/paper/main.pdf.zip)

<!-- Pytest Coverage Comment:Begin -->
\n<!-- Pytest Coverage Comment:End -->


<p align="center">
  <img width = "450" src="src/tess_atlas/webbuilder/template/_static/atlas_logo.png" />
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

### Running analyses
#### Local Run
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

#### Slurm Run

To make the slurm files needed to analyse a CSV of TOIs you can run:
```bash
make_slurm_job --toi_csv toi_ids.csv --module_loads "git/2.18.0 gcc/9.2.0 openmpi/4.0.2 python/3.8.5"
```

Or, if you want to make the slurm job for just one TOI:
```bash
make_slurm_job --toi_number 174 --module_loads 'git/2.18.0 gcc/9.2.0 openmpi/4.0.2 python/3.8.5'
```

### Downloading results
You can download completed analyses with
```bash
download_toi 103 --outdir analysed_tois
```

## Running tests!
Use the following to run tests (skipping slow tests)
```bash
python -m pip install -e ".[test]"
pytest tests/
```
The following only runs the slow ones
```bash
pytest -vv -k "slow" tests/test_template_notebook.py
```

## Building + deploying the catalog
### Building website
Once all your analyses are complete, you can package all the runs into a website:
```bash
make_webpages --webdir webpages --notebooks {notebook_dir} --add-api
```
Using `add-api` will copy over the data files in addition to making the webpages (but can be a bit slow!)
When this completes, you should have a zipped file with the webpages+data: `tess_atlas_pages.tar.gz`

### Deploy website + api data
We are storing the website data on a Nectar project.
Assuming you are a part of the project, the steps to deploy are
1. ssh into Nectar
2. Delete old pages
3. scp `tess_atlas_pages.tar.gz` into Nectar's webdir.
4. untar webpages
```bash
ssh -i ~/.ssh/nectarkey.pem ec2-user@136.186.108.96
cd /mnt/storage/
mv _build trash
scp avajpeyi@ozstar.swin.edu.au:/fred/oz200/avajpeyi/projects/atlas_runs/tess_atlas_pages.tar.gz .
tar -xvzf tess_atlas_pages.tar.gz
rm -rf trash
```


## Publishing `tess_atlas` to pypi
To publish to pypi, you will need admin access to this repo.
Then, publishing just requires you to change the version number `tag` a commit.
The `pypi_release` github action will (hopefully) take care of the rest.
