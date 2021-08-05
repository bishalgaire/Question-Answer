from typing import Optional

from fastapi import FastAPI
from fastapi.logger import logger

from model import Query
from utils import *

#logger.setLevel(logging.DEBUG)


app = FastAPI()

@app.get("/")
async def home():
    return {"app": "QA"}

@app.post("/predict")
async def predict(query: Query):
    logger.info("Received question")
    logger.info(f"Query: {query}")
    if query.query is not None:
        response = pipe.run(query=query.query, top_k_retriever=10, top_k_reader=5)
        logger.info(f"Response: {response}")
        return response
