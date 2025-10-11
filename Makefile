.PHONY: etl baselines lstm all clean

PY=python

etl:
	$(PY) src/etl_spark.py

baselines:
	$(PY) src/train_baselines.py

lstm:
	$(PY) src/train_lstm.py
        #\tpython src/train_lstm.py --seq_len 5 --hidden 64 --layers 1 --dropout 0.1 --epochs 60 --batch_size 128 --lr 1e-3 --patience 10

inference-baseline:
	$(PY) src/inference.py --model baseline

inference-lstm:
	$(PY) src/inference.py --model lstm --serial-id $(SID)

all: etl baselines lstm

clean:
	rm -rf outputs/* __pycache__ .pytest_cache

template:
	$(PY) src/make_template_row.py
