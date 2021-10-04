.PHONY: results tables plots ids data
SHELL:=/bin/bash
DATA_DIR:=./data/
STATS_DIR:=./stats/
CACHE_DIR:=./cache/
JSON_DIR:=./json/
RUNS_DIR:=./runs/
NEURON_DIR:=${DATA_DIR}/neurons/
PDP_DIR:=${DATA_DIR}/pdp/
PYCACHE_DIR:=./classes/__pycache__
# ---------- RUN CONFIGURATION ----------
DATASET:=./data/flows.pickle
RANDOM_SEED:=620
BATCH_SIZE:=128
TRAINING_EPOCHS:=20
VALIDATION_EPOCHS:=${TRAINING_EPOCHS}
PRETRAINING_EPOCHS:=10
TRAINING_PROMILL:=100
PRETRAINING_PROMILL:=800
SUBSET_FILE:=./subsets/10_flows.json
#SUBSET_PARAMETERS:=-G ${SUBSET_FILE}
SUBSET_PARAMETERS:=
BENIGN_CATEGORY:=10

# ---------------------------------------
# Options: PREDICT ID AUTO OBSCURE MASK COMPOSITE
LSTM_PROXY_TASKS:= AUTO PREDICT ID MASK OBSCURE
# Oprions: MASK AUTO OBSCURE
TRANSFORMER_PROXY_TASKS:= MASK AUTO
PDP_FILE:=./data/flows_pdp.json
NEURON_FILE:=./data/flows_neurons.json
PRETRAINING_PARAMETERS:=-s ${PRETRAINING_PROMILL} -E ${PRETRAINING_EPOCHS}
TRAINING_PARAMETERS:=-p ${TRAINING_PROMILL} -e ${TRAINING_EPOCHS} -V ${VALIDATION_EPOCHS} --random_seed ${RANDOM_SEED} -b ${BATCH_SIZE} -c ${BENIGN_CATEGORY}


PDP_PARAMETERS:=-P ${PDP_FILE}
#PDP_PARAMETERS:=
NEURON_PARAMETERS:=-N ${NEURON_FILE}
ID_TMP_FILE:=${CACHE_DIR}/ids_tmp.txt
TMP_FILE:=${CACHE_DIR}/tmp.txt
RESULT_DIR:=./results/${DATASET}_rn${RANDOM_SEED}_1percent/

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*
	rm ${JSON_DIR}/*

lstm:
	python3 main_lstm.py -f ${DATASET}

transformer:
	python3 main_trans.py -f ${DATASET}

cycle: lstm_cycle transformer_cycle
	
test_cycle: lstm_test_cycle transformer_test_cycle
	echo 'Everything seems to work fine'

lstm_cycle:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS}

transformer_cycle:
	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS}
	for pretraining in ${TRANSFORMER_PROXY_TASKS} ; do \
    	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done

lstm_test_cycle:
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${NEURON_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -d

transformer_test_cycle:
	for pretraining in ${TRANSFORMER_PROXY_TASKS} ; do \
    	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_trans.py -f ${DATASET} ${TRAINING_PARAMETERS} -d

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
	python3 plot_pdp.py -f ${PDP_FILE} -D ${PDP_DIR} -i \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp0_xyNONE_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyAUTO_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyBIAUTO_subset|10_flows' \
	'lstm_flows_rn557_hs512_nl3_bs512_ep600_lr001_tp100_sp800_xyPREDICT_subset|10_flows'

neurons:
	python3 plot_neurons.py -f ${NEURON_FILE} -D ${NEURON_DIR} -i \
	'lstm_flows_rn601_hs512_nl3_bs128_tep601_sep0_lr001_tp100_sp0_xyNONE_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep601_sep10_lr001_tp100_sp800_xyID_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyMASK_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyOBSCURE_subset|10_flows' \
	'lstm_flows_rn601_hs512_nl3_bs128_tep600_sep10_lr001_tp100_sp800_xyPREDICT_subset|10_flows'

tmp:
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS}


# --- RESULTS ---
data:
	rm -r ${RESULT_DIR}/stats/ -f
	mkdir -p ${RESULT_DIR}
	mkdir -p ${RESULT_DIR}/stats/
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
		python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${PDP_PARAMETERS} ${NEURON_PARAMETERS} -y $$pretraining -S ${RESULT_DIR}/stats/; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${PDP_PARAMETERS} ${NEURON_PARAMETERS} -S ${RESULT_DIR}/stats/

ids: data
	for pretraining in ${LSTM_PROXY_TASKS} ; do \
		sep=" "; \
		tmp=$$(python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${PRETRAINING_PARAMETERS} ${SUBSET_PARAMETERS} -y $$pretraining --id_only) ; \
		ids=$$ids$$sep$$tmp ; \
		echo $$ids > ${ID_TMP_FILE} ; \
	done
	python3 main_lstm.py -f ${DATASET} ${TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} --id_only | cat - ${ID_TMP_FILE} > ${TMP_FILE} && mv ${TMP_FILE} ${ID_TMP_FILE}

plots: ids
	rm -r ${RESULT_DIR}/neurons/ -f
	rm -r ${RESULT_DIR}/pdp/ -f
	mkdir -p ${RESULT_DIR}/neurons/
	mkdir -p ${RESULT_DIR}/pdp/
#	python3 plot_neurons.py -f ${NEURON_FILE} -D ${NEURON_DIR} -i $$(cat ${ID_TMP_FILE}) -O ${RESULT_DIR}/neurons/
	python3 plot_neurons.py -f ${NEURON_FILE} -D ${NEURON_DIR} -i $$(cat ${ID_TMP_FILE}) -O ${RESULT_DIR}/neurons/ -p pre
#	python3 plot_pdp.py -f ${PDP_FILE} -D ${PDP_DIR} -i $$(cat ${ID_TMP_FILE}) -O ${RESULT_DIR}/pdp/

tables: ids
	rm -r ${RESULT_DIR}/tables/ -f
	mkdir -p ${RESULT_DIR}/tables/
	python3 ./script/tables.py -D ${RESULT_DIR}/stats/ -O ${RESULT_DIR}/tables/
	$(eval OUT_FILES := $(shell python3 ./script/tables.py -D ${RESULT_DIR}/stats/ -O ${RESULT_DIR}/tables/))
	for out_f in $(OUT_FILES) ; do \
		out=$$(echo $$out_f | cut -d'/' -f 7 | cut -d'.' -f 1) ; \
		echo $$out ; \
    	python3 ./script/tably.py $$out_f > ${RESULT_DIR}/tables/$$out.tex ; \
	done

results: tables plots