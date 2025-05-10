
PROJECT_NAME = chathist
PROJECT_VERSION = 0.0.1
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.12


create_venv:
	${PYTHON_INTERPRETER}${PYTHON_VERSION} -m venv venv

install:
	${PYTHON_INTERPRETER}${PYTHON_VERSION}  --version
	pip3 install .

pipreqs:
	pipreqs . --force --scan-notebooks --savepath requirements/requirements.txt

ruff:
	ruff check .