from typing import Callable, Literal, Generator
from functools import wraps
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chathist import Model
import chathist
import json


class PromptInput(BaseModel):
    """
    Enforces the structure of the input for the chat history and title models.
    This class defines the expected input format for the FastAPI endpoints.
    """

    authorized: str | None = None
    prompt: str


app = FastAPI()

chathist.config.load_config(config_path="conf/train", config_name="chat_title")
chat_title_model = Model()


@app.post("/api/v1/title")
def query_title(request: PromptInput) -> StreamingResponse:
    """
    This function handles the query for the chat history model.
    It takes a prompt as input and returns the generated response from the model.

    :param request: The request object containing the prompt to be processed.
    :return: A StreamingResponse that yields the generated tokens from the model.
    It streams the response in JSON format, allowing for real-time updates.
    """

    def stream(prompt: str) -> Generator[str, None, None]:
        try:
            for token in chat_title_model.generate(prompt):
                if token == chathist.config.endoftext_decoded:
                    yield json.dumps(
                        {"status": "success", "response": "", "done": True}
                    ) + "\n"
                else:
                    yield json.dumps(
                        {"status": "success", "response": token, "done": False}
                    ) + "\n"

        except Exception as e:
            print(e)
            yield json.dumps({"status": "faliure", "message": "Internal Error"}) + "\n"

    return StreamingResponse(stream(request.prompt), media_type="application/json")
