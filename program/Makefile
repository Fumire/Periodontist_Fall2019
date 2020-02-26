# Variables
RANDOM = $(shell bash -c 'echo $$RANDOM')
JOBS = 10
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
	$(PYTHON3) $(abspath $(filter %.py,$^)) --xlsx --output_file $@

$(RESULTS_DIRECTORY)/csv/default.%.csv: data.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --number --input_file $(filter %.csv,$^) --output_file $@

# tsne
$(RESULTS_DIRECTORY)/tsne/%.csv: tsne.py $(RESULTS_DIRECTORY)/csv/%.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --calculate --input_file $(filter %.csv,$^) --output_file $@

$(RESULTS_DIRECTORY)/tsne/%.png: tsne.py $(RESULTS_DIRECTORY)/tsne/%.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --draw --input_file $(filter %.csv,$^) --output_file $@

# Multiclass
$(RESULTS_DIRECTORY)/5-class/statistics.csv: multiclass_classification.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --five --input_file $(filter %.csv,$^) --output_dir $(dir $@) --jobs $(JOBS)

$(RESULTS_DIRECTORY)/4-class/statistics.csv: multiclass_classification.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --four --input_file $(filter %.csv,$^) --output_dir $(dir $@) --jobs $(JOBS)

$(RESULTS_DIRECTORY)/3-class/statistics.csv: multiclass_classification.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --three --input_file $(filter %.csv,$^) --output_dir $(dir $@) --jobs $(JOBS)

# Two-class
$(RESULTS_DIRECTORY)/2-class/one-vs-one/statistics.csv: biclass_classification.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --ovo --input_file $(filter %.csv,$^) --output_dir $(dir $@) --jobs $(JOBS)

$(RESULTS_DIRECTORY)/2-class/one-vs-rest/statistics.csv: biclass_classification.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --ovr --input_file $(filter %.csv,$^) --output_dir $(dir $@) --jobs $(JOBS)

# Classification Plot
$(RESULTS_DIRECTORY)/3-class/plot: draw_classification.py $(RESULTS_DIRECTORY)/3-class/statistics.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --stat --input_file $(filter %.csv,$^) --output_dir $@

# Draw about depth
$(RESULTS_DIRECTORY)/depth/scatter.png: draw_depth.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --scatter --input_file $(filter %.csv,$^) --output_file $@

$(RESULTS_DIRECTORY)/depth/violin_AL.png: draw_depth.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --violin --input_file $(filter %.csv,$^) --output_file $@ --column AL

$(RESULTS_DIRECTORY)/depth/violin_PD.png: draw_depth.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --violin --input_file $(filter %.csv,$^) --output_file $@ --column PD

# Regressor
$(RESULTS_DIRECTORY)/Regression-AL/statistics.csv: regression.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --input_file $(filter %.csv,$^) --output_dir $(dir $@) --column AL --jobs $(JOBS)

$(RESULTS_DIRECTORY)/Regression-PD/statistics.csv: regression.py $(RESULTS_DIRECTORY)/csv/default.csv
	$(PYTHON3) $(abspath $(filter %.py,$^)) --input_file $(filter %.csv,$^) --output_dir $(dir $@) --column PD --jobs $(JOBS)

latest: $(RESULTS_DIRECTORY)/Regression-AL/statistics.csv
