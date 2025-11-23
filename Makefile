activate:
	source .venv/bin/activate

install:
	pip install -r requirements.txt

train:
	python training_harness.py

predict:
	python predict_harness.py

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf data/outputs
	rm -rf data/inputs