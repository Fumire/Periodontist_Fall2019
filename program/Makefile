# Variables
RANDOM = $(shell bash -c 'echo $$RANDOM')
THREADS = 32
PWD = $(shell pwd)

all:
	if [ ! -d $(PWD)/log ]; then mkdir $(PWD)/log; fi;
	echo "make -C $(PWD) absolute_real relative_real both_real" > tmp.sh
.PHONY += all

clean:
	rm -rfv test
	rm -rfv PNG
.PHONY += clean

start: tmp.sh
	qsub -N Perio -pe smp $(THREADS) -o $(PWD)/log/$(ID).stdout -e $(PWD)/log/$(ID).stderr $<
.PHONY += start

absolute_real: 730.py
	python3 $(abspath $<) --pickle_dir "results/absolute" --jobs $(THREADS) --absolute

absolute_test: 730.py
	python3 $(abspath $<) --verbose --pickle_dir "test/$(RANDOM)" --jobs $(THREADS) --absolute

relative_real: 730.py
	python3 $(abspath $<) --pickle_dir "results/relative" --jobs $(THREADS) --relative

relative_test: 730.py
	python3 $(abspath $<) --verbose --pickle_dir "test/$(RANDOM)" --jobs $(THREADS) --relative

both_real: 730.py
	python3 $(abspath $<) --pickle_dir "results/both" --jobs $(THREADS) --both

both_test: 730.py
	pyton3 $(abspath $<) --verbose --pickle_dir "test/$(RANDOM)" --jobs $(THREADS) --both
