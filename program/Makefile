# Variables
RANDOM = $(shell bash -c 'echo $$RANDOM')
THREADS = 32
PWD = $(shell pwd)

# Tools
PYTHON3 = /BiO/Live/jwlee230/.pyenv/shims/python3

# Directory
RESULTS_DIRECTORY = /BiO/Research/Periodontist_Fall2019/results

all:
	if [ ! -d $(PWD)/log ]; then mkdir $(PWD)/log; fi;
	echo "make -C $(PWD) absolute_real relative_real both_real" > tmp.sh
.PHONY += all

clean:
	rm -rfv log
	rm -rfv __pycache__
.PHONY += clean

start: tmp.sh
	if [ ! -d $(PWD)/log ]; then mkdir $(PWD)/log; fi;
	qsub -N Perio -pe smp $(THREADS) -o $(PWD)/log/stdout -e $(PWD)/log/stderr $<
.PHONY += start

# data
$(RESULTS_DIRECTORY)/csv/default.csv: data.py
	$(PYTHON3) $(abspath $<) --xlsx --output_file $@

$(RESULTS_DIRECTORY)/csv/%.remove_ap.csv: data.py $(RESULTS_DIRECTORY)/csv/%.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --remove_ap --input_file $(filter %.csv, $^) --output_file $@

# tsne
$(RESULTS_DIRECTORY)/tsne/%.csv: tsne.py $(RESULTS_DIRECTORY)/csv/%.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --calculate --input_file $(filter %.csv, $^) --output_file $@

$(RESULTS_DIRECTORY)/tsne/%.png: tsne.py $(RESULTS_DIRECTORY)/tsne/%.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --draw --input_file $(filter %.csv, $^) --output_file $@

latest: $(RESULTS_DIRECTORY)/tsne/default.png
