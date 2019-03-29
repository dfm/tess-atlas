VERSION=0.1.1
FILES=$(wildcard notebooks/${VERSION}/*.ipynb)
NOTEBOOKS = $(foreach fn, $(FILES), docs/$(basename ${fn}).html)

docs/notebooks/${VERSION}/%.html: notebooks/${VERSION}/%.ipynb toipage.tpl
	jupyter nbconvert --to html $< --output-dir docs/notebooks/${VERSION} --template=toipage.tpl

html: $(NOTEBOOKS)
	python collect_notebooks.py ${VERSION}
	git add docs/notebooks/${VERSION}/*.html

website:
	rm -rf build
	git clone . build
	cd build
	make html
	git checkout --orphan gh-pages
	git rm --cached -rf .
	cp -R docs/* .
	cp docs/.nojekyll .
	git add -f CNAME css static index.html notebooks .nojekyll
	git commit -m "updating the site"
	git push -f origin gh-pages

.PHONY: html website
default: html
