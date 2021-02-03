DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
DATASET:=CAIA
STATS_DIR:=./stats
CACHE_DIR:=./cache
JSON_DIR:=./json
PYCACHE_DIR:=./classes/__pycache__

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*
	rm ${JSON_DIR}/*

test:
	python3 main_lstm.py -f ./data/flows.pickle -b 128 -e 5 -g -s 90 --no_cache -x PREDICT