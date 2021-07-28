DATA_DIR:=./data/rn601/
SHELL:=/bin/bash
TEX_DIR:=./tex/

all:
	$(eval OUT_FILES := $(shell python3 ./tables.py -D ${DATA_DIR}))
	for out_f in $(OUT_FILES) ; do \
		out=$$(echo $$out_f | cut -d'/' -f 3 | cut -d'.' -f 1) ; \
    	python3 tably.py $$out_f > ${TEX_DIR}$$out.tex ; \
	done

class:
	$(eval OUT_FILES := $(shell python3 ./tables.py -D ${DATA_DIR} -m CLASS))
	for out_f in $(OUT_FILES) ; do \
		out=$$(echo $$out_f | cut -d'/' -f 3 | cut -d'.' -f 1) ; \
    	python3 tably.py $$out_f > ${TEX_DIR}$$out.tex ; \
	done

stats:
	$(eval OUT_FILES := $(shell python3 ./tables.py -D ${DATA_DIR} -m STATS))
	for out_f in $(OUT_FILES) ; do \
		out=$$(echo $$out_f | cut -d'/' -f 3 | cut -d'.' -f 1) ; \
    	python3 tably.py $$out_f > ${TEX_DIR}$$out.tex ; \
	done

