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
SUBSET_FILE=single_category_cic2017_ratio.json
CYCLE_PRETRAINING_PARAMETERS:=-s 800 -E 10 -V 10
CYCLE_TRAINING_PARAMETERS:=-p 100 -e 100 -V 10 --random_seed 556 -G ./subsets/${SUBSET_FILE} -b 256
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
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} -d

transformer_test_cycle:
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_trans.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} -d

lstm_single_category:
	for index in 0 1 2 3 4 5 6 7 9 10 11 13 ; do \
    	python3 main_lstm.py -f ${DATASET} ${CYCLE_TRAINING_PARAMETERS} ${CYCLE_PRETRAINING_PARAMETERS} -y PREDICT -i $$index ; \
	done