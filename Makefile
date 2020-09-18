VERSION=0.1.1
FILES=$(wildcard notebooks/${VERSION}/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

docs/notebooks/${VERSION}/%.html: notebooks/${VERSION}/%.ipynb tess_atlas/toipage.tpl
	jupyter nbconvert --to html $< --output-dir docs/notebooks/${VERSION} --template=tess_atlas/toipage.tpl --config=tess_atlas/cfg.py

html: $(NOTEBOOKS)
	python tess_atlas/collect_notebooks.py ${VERSION}

website:
	./build_website.sh

.PHONY: html website
default: html
