
PROJECT_NAME = sculpgen
PROJECT_VERSION = 0.0.1
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.12


create_venv:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} --version
	${PYTHON_INTERPRETER}${PYTHON_VERSION} -m venv venv

install:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} --version
	pip3 install .

install_dev:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} --version
	pip3 install .[dev]


pipreqs:
	pipreqs . --force --scan-notebooks --savepath requirements/requirements.txt

ruff:
	ruff check .


##### data related commands
images:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} ${PROJECT_NAME}/data/__init__.py

model:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} ${PROJECT_NAME}/model.py