version = $(shell cat package.json | grep version | awk -F'"' '{print $$4}')

install:
	pip3 install poetry black isort
	poetry install

run:
	poetry run python3 main.py

lint:
	isort *.py
	black *.py

train_vgg19:
	poetry run python3 train_vgg19.py

train_resnet50:
	poetry run python3 train_resnet50.py
