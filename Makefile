# vim:set ft=make ts=4 sw=4 sts=4 autoindent:

# A simple non-GNU Makefile to build nerv.
#
# Author:		Pontus Stenetorp	<pontus stenetorp se>
# Version:		2014-03-11

CWD=`pwd`

all:
	cd src/nerv/ && ./setup.py build_ext --inplace

.PHONY: sanity
sanity:
	for m in dag maths net; \
	do \
		PYTHONPATH="${CWD}/src" test/sanity/$${m}.py; \
	done;

.PHONY: perf
perf:
	for m in maths net; \
	do \
		PYTHONPATH="${CWD}/src" test/perf/$${m}.py; \
	done;

.PHONY: clean
clean:
	find . -type d -a -name __pycache__ | xargs -r rm -r -f
	cd src/nerv/ && ./setup.py clean
	for d in `find . -type d -a -name cy`; \
	do \
		find $${d} -name '*.c' | xargs -r rm; \
	done;
	find src -name '*.so' | xargs -r rm
