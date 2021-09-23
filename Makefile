VERSION_FILE='./tess_atlas/tess_atlas_version.py'
VERSION:=$$(grep -Eo '[0-9]+\.[0-9]+\.[0-9]' ${VERSION_FILE})

website:
	./build_website.sh

run:
	python tess_atlas/run_tois.py

test:
	pytest -vv -k "slow" tests/test_template_notebook.py

.PHONY: html website
default: html
