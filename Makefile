VERSION="0.1.0"
FILES=$(wildcard notebooks/${VERSION}/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

docs/notebooks/${VERSION}/%.html: notebooks/${VERSION}/%.ipynb
	jupyter nbconvert --to html $< --output-dir docs/notebooks/${VERSION}

html: $(NOTEBOOKS)
	echo $(FILES)

.PHONY: html
