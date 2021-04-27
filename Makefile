DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
STATS_DIR:=./stats
CACHE_DIR:=./cache
JSON_DIR:=./json
RUNS_DIR:=./runs
PYCACHE_DIR:=./classes/__pycache__
LSTM_PRETRAININGS:=OBSCURE MASK PREDICT
TRANSFORMER_PRETRAININGS:=MASK AUTO
CYCLE_PRETRAINING_PARAMETERS:=-p 1 -s 89 -E 10 -e 101 --no_cache
CYCLE_TRAINING_PARAMETERS:=-p 1 -e 101 --no_cache

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*
	rm ${JSON_DIR}/*

lstm:
	python3 main_lstm.py -f ./data/flows.pickle

transformer:
	python3 main_trans.py -f ./data/flows.pickle

cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS}
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS}
	

test_cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	echo 'Everything seems to work fine'
	
lstm_cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS}

transformer_cycle:
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining ; \
	done
	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS}

lstm_test_cycle:
	for pretraining in ${LSTM_PRETRAININGS} ; do \
    	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_lstm.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS} -d

transformer_test_cycle:
	for pretraining in ${TRANSFORMER_PRETRAININGS} ; do \
    	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_PRETRAINING_PARAMETERS} -y $$pretraining -d ; \
	done
	python3 main_trans.py -f ./data/flows.pickle ${CYCLE_TRAINING_PARAMETERS} -d
