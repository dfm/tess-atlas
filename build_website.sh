#!/bin/sh

rm -rf build
git clone . build
cd build
pwd
make html
git checkout --orphan gh-pages
git rm --cached -rf .
cp -R docs/* .
cp docs/.nojekyll .
git add -f CNAME css static index.html notebooks .nojekyll
git commit -m "updating the site"
git push -f https://github.com/dfm/tess-atlas.git gh-pages
