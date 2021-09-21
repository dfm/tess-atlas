TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, atlas.${suff}))
CHECK_RERUN = grep Rerun atlas.log
TECTONIC    = $(shell command -v tectonic >/dev/null && echo true || echo false )

default: atlas.pdf

atlas.pdf: atlas.tex atlas.bib dfmstyle.tex dfmsymbols.tex
	# Generate links to current git commit
	python gen_links.py
	if [ "${TECTONIC}" = "true" ]; then\
		tectonic atlas.tex --print --keep-logs;\
	else\
		${LATEX} atlas;\
		bibtex atlas;\
		( ${CHECK_RERUN} && ${LATEX} atlas ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} atlas ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} atlas ) || echo "Done.";\
	fi

clean:
	$(RM_TMP)

.PHONY: clean