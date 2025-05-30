FROM python:3.12-slim

WORKDIR /app


COPY requirements requirements
COPY pyproject.toml pyproject.toml
COPY chathist chathist
COPY models/finetuned/gpt2-xl_indian_history_finetuned models/finetuned/gpt2-xl_indian_history_finetuned
COPY server server

RUN pip3 install . --no-cache
RUN pip3 install ."[dev]" --no-cache

ENTRYPOINT ["sh", "-c", "uvicorn server.chat_history:app --reload" ]