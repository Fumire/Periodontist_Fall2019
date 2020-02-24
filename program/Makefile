# Variables
RANDOM = $(shell bash -c 'echo $$RANDOM')
THREADS = 32
PWD = $(shell pwd)

# Tools
PYTHON3 = /BiO/Live/jwlee230/.pyenv/shims/python3

# Directory
RESULTS_DIRECTORY = /BiO/Research/Periodontist_Fall2019/results

all:
.PHONY += all

clean:
	rm -rfv log
	rm -rfv __pycache__
.PHONY += clean

start: tmp.sh
.PHONY += start

# data
$(RESULTS_DIRECTORY)/csv/default.csv: data.py
	$(PYTHON3) $(abspath $<) --xlsx --output_file $@

$(RESULTS_DIRECTORY)/csv/default.%.csv: data.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --number --input_file $(filter %.csv, $^) --output_file $@

# tsne
$(RESULTS_DIRECTORY)/tsne/%.csv: tsne.py $(RESULTS_DIRECTORY)/csv/%.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --calculate --input_file $(filter %.csv, $^) --output_file $@

$(RESULTS_DIRECTORY)/tsne/%.png: tsne.py $(RESULTS_DIRECTORY)/tsne/%.csv
	$(PYTHON3) $(abspath $(filter %.py, $^)) --draw --input_file $(filter %.csv, $^) --output_file $@

latest: $(RESULTS_DIRECTORY)/tsne/default.png
