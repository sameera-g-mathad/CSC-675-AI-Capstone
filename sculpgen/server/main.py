from typing import Literal
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount(
    "/images",
    StaticFiles(directory="/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/nst"),
    name="images",
)

# class ResponseData(BaseModel):
#     status: Literal["error", "success"]


# class Response(BaseModel):
#     statusCode: int
#     data: ResponseData
