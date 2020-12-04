DATA_DIR:=~/ucloud/thesis/code/Datasets-preprocessing/CIC-IDS-2017
DATASET:=CAIA
STATS_DIR:=./stats
CACHE_DIR:=./cache
PYCACHE_DIR:=./classes/__pycache__

clean:
	rm ${CACHE_DIR}/*
	rm ${PYCACHE_DIR}/*

test:
	python3 main.py -f ./data/flows.pickle -g -t