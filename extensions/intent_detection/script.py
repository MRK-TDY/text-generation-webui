from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from modules.logging_colors import logger


params = {
    "device": "1"
}


def setup():
    global embedding_model
    device = int(params["device"])
    if torch.cuda.is_available() and device < torch.cuda.device_count():
        device = f"cuda:{device}"
        device = torch.device(device)
        embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
    else:
        embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

def intent_similarity(query: str, intent_samples: list[str], threshold: float = 0.8) -> float:
    global embedding_model
    docs = [query] + intent_samples
    logger.info(f'Query: {query}')
    logger.info(f'Intent samples: {intent_samples}')
    embeddings = embedding_model.encode(docs)
    similarities = cos_sim(embeddings[0], embeddings[1:]).flatten()
    logger.info(f'Similarities: {similarities}')
    return max(similarities)
