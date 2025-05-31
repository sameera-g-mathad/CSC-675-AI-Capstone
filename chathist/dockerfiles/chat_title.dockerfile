FROM python:3.12-slim

WORKDIR /app


COPY requirements requirements
COPY pyproject.toml pyproject.toml
COPY chathist chathist
COPY models/finetuned/gpt2-medium_chat_title_finetuned models/finetuned/gpt2-medium_chat_title_finetuned
COPY server server

RUN pip3 install . --no-cache
RUN pip3 install ."[dev]" --no-cache

ENTRYPOINT ["sh", "-c", "uvicorn server.chat_title:app --host  0.0.0.0 --reload --port 8001" ]