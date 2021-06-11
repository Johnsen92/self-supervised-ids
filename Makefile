DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
STATS_DIR:=./stats
CACHE_DIR:=./cache
JSON_DIR:=./json
RUNS_DIR:=./runs
PYCACHE_DIR:=./classes/__pycache__
# Options: PREDICT AUTO BIAUTO OBSCURE MASK
LSTM_PRETRAININGS:=PREDICT AUTO BIAUTO
# Oprions: MASK AUTO OBSCURE
TRANSFORMER_PRETRAININGS:=MASK AUTO
SUBSET_FILE:=./subsets/10_flows.json
PDP_FILE:=./data/flows_features.json
CYCLE_PRETRAINING_PARAMETERS:=-s 800 -E 10
CYCLE_TRAINING_PARAMETERS:=-p 100 -e 300 -V 10 --random_seed 557 -b 128
SUBSET_PARAMETERS:=-G ${SUBSET_FILE}
PDP_PARAMETERS:=-P ${PDP_FILE}
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
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ${DATASET} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	echo 'Everything seems to work fine'

lstm_cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS}

transformer_cycle:
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS}

lstm_test_cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d --no_cache ; \
	done
	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} -d

transformer_test_cycle:
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d --no_cache ; \
	done
	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} -d

lstm_single_category_pretraining:
	for index in 0 1 2 3 4 5 6 7 9 10 11 13 ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y BIAUTO -i $$index ; \
	done

lstm_single_category:
	for index in 0 1 2 3 4 5 6 7 9 10 11 13 ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} -i $$index ; \
	done

debug:
	python3 main_lstm.py -f ${DATASET} -p 1 -e 1 -V 10 --random_seed 556 -y PREDICT

lstm_pdp:
	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${SUBSET_PARAMETERS} ${PDP_PARAMETERS}

