FROM python3.12-slim

WORKDIR /chathist

COPY /requirements /requirements
COPY pyproject.toml pyproject.toml
COPY /chathist /chathist

COPY /server /server

RUN pip install . --no-cache

ENTRYPOINT [ "uvicorn", "server.main:app", "--reload" ]