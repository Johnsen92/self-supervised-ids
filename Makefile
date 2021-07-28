SHELL:=/bin/bash
DATA_DIR:=./data/
STATS_DIR:=./stats/
CACHE_DIR:=./cache/
JSON_DIR:=./json/
RUNS_DIR:=./runs/
PYCACHE_DIR:=./classes/__pycache__
# Options: PREDICT ID AUTO OBSCURE MASK COMPOSITE
LSTM_PROXY_TASKS:= AUTO ID PREDICT OBSCURE MASK 
# Oprions: MASK AUTO OBSCURE
TRANSFORMER_PROXY_TASKS:= MASK AUTO
SUBSET_FILE:=./subsets/10_flows.json
PDP_FILE:=./data/flows_pdp.json
NEURON_FILE:=./data/flows_neurons.json
PRETRAINING_PARAMETERS:=-s 800 -E 10
TRAINING_PARAMETERS:=-p 100 -e 600 -V 10 --random_seed 601 -b 128
SUBSET_PARAMETERS:=-G ${SUBSET_FILE}
PDP_PARAMETERS:=-P ${PDP_FILE}
NEURON_PARAMETERS:=-N ${NEURON_FILE}
DATASET:=./data/flows.pickle

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*
	rm ${JSON_DIR}/*

lstm:
	python3 main_lstm.py -f ${DATASET}

transformer:
	python3 main_trans.py -f ${DATASET}

cycle: lstm_cycle transformer_cycle
	
test_cycle:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	for pretraining in ${TRANSFORMER_PROXY_TASKS} ; do \
    	python3 main_trans.py -f ${DATASET} ${PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	echo 'Everything seems to work fine'

lstm_cycle:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -y $$pretraining --id_only ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS}

transformer_cycle:
	for pretraining in ${TRANSFORMER_PROXY_TASKS} ; do \
    	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS}

lstm_test_cycle:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${NEURON_PARAMETERS} -y $$pretraining -d --no_cache; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -d --no_cache

transformer_test_cycle:
	for pretraining in ${TRANSFORMER_PROXY_TASKS} ; do \
    	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} -y $$pretraining -d --no_cache ; \
	done
	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} -d

lstm_single_category_pretraining:
	for index in 0 1 2 3 4 5 6 7 9 10 11 13 ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} -y BIAUTO -i $$index ; \
	done

lstm_single_category:
	for index in 0 1 2 3 4 5 6 7 9 10 11 13 ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} -i $$index ; \
	done

debug:
	python3 main_lstm.py -f ${DATASET} -p 1 -e 1 -V 10 --random_seed 556 -y PREDICT

lstm_pdp:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${PDP_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${PDP_PARAMETERS}

lstm_neurons:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${NEURON_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${NEURON_PARAMETERS}

lstm_pdp_debug:
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PDP_PARAMETERS} -d

pdp:
#	python3 plot_pdp.py -f ${PDP_FILE} -D ./data/pdp/ -i 'lstm_flows_hs512_nl3_bs128_ep300_lr001_tp100_sp0_xyNONE_subset|10_flows' 'lstm_flows_hs512_nl3_bs128_ep300_lr001_tp100_sp800_xyAUTO_subset|10_flows' 'lstm_flows_hs512_nl3_bs128_ep300_lr001_tp100_sp800_xyBIAUTO_subset|10_flows' 'lstm_flows_hs512_nl3_bs128_ep300_lr001_tp100_sp800_xyPREDICT_subset|10_flows'
	python3 plot_pdp.py -f ${PDP_FILE} -D ./data/pdp/ -i \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp0_xyNONE_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyAUTO_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyBIAUTO_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyPREDICT_subset|10_flows'

neurons:
	python3 plot_neurons.py -f ${NEURON_FILE} -D ./data/neurons/ -i \
	'lstm_flows_rn601_hs512_nl3_bs128_tep601_sep0_lr001_tp100_sp0_xyNONE_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep601_sep10_lr001_tp100_sp800_xyID_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyMASK_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyOBSCURE_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyPREDICT_subset|10_flows'

tmp:
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} --no_cache

results:
#	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS}
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
		sep=" "; \
		tmp=$$(python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -y $$pretraining --id_only) ; \
		ids=$$ids$$sep$$tmp ; \
#		python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -y $$pretraining ; \
		echo $$ids > ${CACHE_DIR}/ids_tmp.txt ; \
	done
	
	for id in $$(cat ${CACHE_DIR}/ids_tmp.txt) ; do \
		echo $$id ; \
	done
	
	