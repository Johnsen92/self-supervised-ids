DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
STATS_DIR:=./stats
CACHE_DIR:=./cache
JSON_DIR:=./json
RUNS_DIR:=./runs
PYCACHE_DIR:=./classes/__pycache__
LSTM_PRETRAININGS:=OBSCURE MASK PREDICT AUTO
TRANSFORMER_PRETRAININGS:=MASK AUTO
CYCLE_PRETRAINING_PARAMETERS:=-s 899 -E 10 -e 300 -V 25
CYCLE_TRAINING_PARAMETERS:=-p 1 -e 300 -V 25 --no_cache --random_seed 556
DATASET:=./data/flows15.pickle

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
