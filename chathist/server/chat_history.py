from typing import Callable, Literal
from functools import wraps
from fastapi import FastAPI
from pydantic import BaseModel
from chathist import Model
import chathist


class PromptInput(BaseModel):
    """
    Enforces the structure of the input for the chat history and title models.
    This class defines the expected input format for the FastAPI endpoints.
    """

    authorized: str | None = None
    prompt: str


class ResponseData(BaseModel):
    """
    Represents the data structure for the response from the FastAPI endpoints.
    This class defines the expected output format, including the status and response message.
    """

    status: Literal["error", "success"]
    response: str


class Response(BaseModel):
    """
    Represents the standardized response format for the FastAPI endpoints.
    This class encapsulates the status code and the data returned from the endpoints.
    """

    statusCode: int
    data: ResponseData


app = FastAPI()

# set first model - History model
chathist.config.load_config(config_path="conf/train", config_name="history")
chat_hist_model = Model()

# # set second model - Chat title model
# chathist.config.load_config(config_path="conf/train", config_name="chat_title")
# chat_title_model = Model()


def _run(func: Callable):
    """
    A decorator to handle exceptions and return a standardized response format.
    This decorator wraps the function to catch any exceptions that may occur during its execution.
    It returns a Response object with a status code and a message indicating success or failure.

    :param func: The function to be wrapped.
    :return: A Response object containing the status code and data.
    :rtype: Response
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Response:
        try:
            result = func(*args, **kwargs)
            return Response(
                statusCode=200,
                data=ResponseData(status="success", response=result),
            )
        except Exception as e:
            print(e)
            return Response(
                statusCode=500,
                data=ResponseData(status="error", response="Internal error"),
            )

    return wrapper


@app.get("/query", response_model=Response)
@_run
def query_history(request: PromptInput) -> str:
    """
    This function handles the query for the chat history model.
    It takes a prompt as input and returns the generated response from the model.

    :param request: PromptInput object containing the prompt to be processed.
    :return: A dictionary containing the status and the generated response from the model.
    :rtype: dict
    """
    prompt = request.prompt
    return chat_hist_model.generate(prompt=prompt)


# @app.get("/title", response_model=Response)
# @_run
# def query_title(request: PromptInput) -> str:
#     """
#     This function handles the query for the chat title model.
#     It takes a prompt as input and returns the generated response from the model.

#     :param request: PromptInput object containing the prompt to be processed.
#     :return: A dictionary containing the status and the generated response from the model.
#     :rtype: dict
#     """
#     prompt = request.prompt
#     return chat_title_model.generate(prompt=prompt)
