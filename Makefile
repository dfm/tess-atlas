VERSION=0.2.0
FILES=$(wildcard notebooks/${VERSION}/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

docs/notebooks/${VERSION}/%.html: notebooks/${VERSION}/%.ipynb tess_atlas/toipage.tpl
	jupyter nbconvert --to html $< --output-dir docs/notebooks/${VERSION} --template=tess_atlas/toipage.tpl --config=tess_atlas/cfg.py

html: $(NOTEBOOKS)
	python tess_atlas/make_tois_homepage.py ${VERSION}

website:
	./build_website.sh

run:
	python tess_atlas/run_tois.py

.PHONY: html website
default: html
