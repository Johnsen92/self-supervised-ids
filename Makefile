DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
STATS_DIR:=./stats
CACHE_DIR:=./cache
JSON_DIR:=./json
RUNS_DIR:=./runs
PYCACHE_DIR:=./classes/__pycache__

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*
	rm ${JSON_DIR}/*

lstm:
	python3 main_lstm.py -f ./data/flows.pickle

transformer:
	python3 main_trans.py -f ./data/flows.pickle

cycle:
	python3 main_lstm.py -f ./data/flows.pickle -y OBSCURE -p 1 -s 89 --no_cache
	python3 main_lstm.py -f ./data/flows.pickle -y MASK -p 1 -s 89 --no_cache
	python3 main_lstm.py -f ./data/flows.pickle -y PREDICT -p 1 -s 89 --no_cache
	python3 main_lstm.py -f ./data/flows.pickle -p 1 --no_cache
	python3 main_trans.py -f ./data/flows.pickle -y MASK -p 1 -s 89 --no_cache
	python3 main_trans.py -f ./data/flows.pickle -y AUTO -p 1 -s 89 --no_cache
	python3 main_trans.py -f ./data/flows.pickle -p 1 --no_cache