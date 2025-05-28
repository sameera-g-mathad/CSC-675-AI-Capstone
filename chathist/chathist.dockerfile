FROM python:3.12-slim

WORKDIR /app

COPY requirements requirements
COPY pyproject.toml pyproject.toml
COPY chathist chathist
COPY models/finetuned models/finetuned
COPY server server

RUN pip3 install . --no-cache
RUN pip3 install ."[dev]" --no-cache

ENTRYPOINT [ "uvicorn", "server.main:app", "--reload" ]