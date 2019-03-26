FILES=$(wildcard notebooks/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

build/rst/notebooks/%.rst: notebooks/%.ipynb notebooks/toi_rst.tpl
	jupyter nbconvert --template notebooks/toi_rst --to rst $< --output-dir build/rst/notebooks

docs/notebooks/%.html: notebooks/%.ipynb
	jupyter nbconvert --to html $< --output-dir docs/notebooks

html: $(NOTEBOOKS)
	echo $(FILES)

.PHONY: html
