from sentence_transformers.util import cos_sim
import torch
from modules.logging_colors import logger
import numpy as np
import httpx
import traceback


params = {
    "endpoint": ""
}


def setup():
    return


async def intent_similarity(query: str, intent_samples: list[str], threshold: float = 0.8) -> float:
    docs = [query] + intent_samples
    logger.info(f'Query: {query}')
    logger.info(f'Intent samples: {intent_samples}')
    async with httpx.AsyncClient() as client:
        response = await client.post(params["endpoint"], json=docs)
        embeddings = response.json()
    embeddings = torch.tensor(embeddings)
    similarities = cos_sim(embeddings[0], embeddings[1:]).flatten()
    logger.info(f'Similarities: {similarities}')
    return max(similarities)
