VERSION=0.1.1
FILES=$(wildcard notebooks/${VERSION}/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

docs/notebooks/${VERSION}/%.html: notebooks/${VERSION}/%.ipynb toipage.tpl
	jupyter nbconvert --to html $< --output-dir docs/notebooks/${VERSION} --template=toipage.tpl

html: $(NOTEBOOKS)
	python collect_notebooks.py ${VERSION}
	git add docs/notebooks/${VERSION}/*.html

website:
	./build_website.sh

.PHONY: html website
default: html
