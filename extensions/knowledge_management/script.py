import requests
import json
from modules.logging_colors import logger


params = {
    "knowledge_api": ""
}


def get_context(user_input: str, history: list, filters: list, top_k: int = 5, history_n: int = 2):
    data = {
        "query": user_input,
        "history": history,
        "filters": {
            "filter_key": filters
        },
        "top_k": top_k,
        "history_n": history_n
    }

    results = requests.get(f"{params['knowledge_api']}/search", json=data)
    results = results.json()
    results = results["results"]
    if len(results) == 0:
        return ""
    results = "\n".join(results)
    context = f"\n<knowledge_start>\n{results}\n<knowledge_end>\n"
    return context

