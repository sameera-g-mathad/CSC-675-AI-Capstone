from typing import Callable, Literal, Generator
from functools import wraps
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from chathist import Model
import chathist


class PromptInput(BaseModel):
    """
    Enforces the structure of the input for the chat history and title models.
    This class defines the expected input format for the FastAPI endpoints.
    """

    authorized: str | None = None
    prompt: str


app = FastAPI()

# set first model - History model
chathist.config.load_config(config_path="conf/train", config_name="history")
chat_hist_model = Model()


@app.post("/api/v1/query")
def query_history(request: PromptInput) -> StreamingResponse:
    """
    This function handles the query for the chat history model.
    It takes a prompt as input and returns the generated response from the model.

    :param request: The request object containing the prompt to be processed.
    :return: A StreamingResponse that yields the generated tokens from the model.
    It streams the response in JSON format, allowing for real-time updates.
    """

    def stream() -> Generator[str, None, None]:
        try:
            prompt = request.prompt
            for token in chat_hist_model.generate(prompt):
                if token == chathist.config.endoftext:
                    yield json.dumps(
                        {"status": "success", "response": "", "done": True}
                    ) + "\n"
                    return
                yield json.dumps(
                    {"status": "success", "response": token, "done": False}
                ) + "\n"
        except Exception as e:
            print(e)
            yield json.dumps(
                {"status": "faliure", "response": "Internal Error", "done": True}
            ) + "\n"

    return StreamingResponse(stream(), media_type="application/json")
