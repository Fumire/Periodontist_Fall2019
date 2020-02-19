# Variables
RANDOM = $(shell bash -c 'echo $$RANDOM')

all:
.PHONY += all

clean:
	rm -rfv pickle
	rm -rfv PNG
.PHONY += clean

absolute: 730.py
	python3 $(abspath $<) --verbose --pickle_dir "pickle/$(RANDOM)" --absolute
