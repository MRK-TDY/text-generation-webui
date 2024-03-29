import requests
import re
from modules.logging_colors import logger


params = {
    "knowledge_api_endpoint": ""
}


def get_context(user_input: str, history: list, filters: list, top_k: int = 10, history_n: int = 1):
    data = {
        "query": user_input,
        "history": history,
        "filters": {
            "filter_key": filters
        },
        "top_k": top_k,
        "history_n": history_n
    }
    try:
        results = requests.get(f"{params['knowledge_api_endpoint']}/search", json=data)
        results = results.json()
        results = results["results"]
    except Exception as e:
        logger.error(e)
        return ""

    if len(results) == 0:
        return ""
    results = "\n".join(results)
    context = f"\n{results}\n"
    return context


def input_modifier(string, state, is_chat=False):
    history = state['history']['internal']
    history = [message for dialogue_round in history for message in dialogue_round] if len(history) > 0 else []
    knowledge = get_context(string, history, ["world", state["name2"]])
    state['context'] = state['context'].replace('<knowledge_injection>', f'{knowledge}')
    return string
