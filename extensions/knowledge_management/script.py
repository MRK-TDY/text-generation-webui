import requests
import httpx
import re
from modules.logging_colors import logger


params = {
    "knowledge_api_endpoint": ""
}


async def get_context(user_input: str, history: list, filters: list, top_k: int = 6, history_n: int = 1, score_threshold=0.45):
    data = {
        "query": user_input,
        "history": history,
        "filters": {
            "filter_key": filters
        },
        "top_k": top_k,
        "history_n": history_n,
        "score_threshold": score_threshold
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{params['knowledge_api_endpoint']}/search", json=data)
            results = response.json()
            results = results["results"]
    except Exception as e:
        logger.error(e)
        return ""

    if len(results) == 0:
        return ""
    results = [f'Chunk {i+1}: {result}' for i, result in enumerate(results)]
    results = "\n".join(results)
    context = f"\n{results}\n"
    return context


async def add_memory(entries):
    data = {
        "entries": entries
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{params['knowledge_api_endpoint']}/add", json=data)

    return response
