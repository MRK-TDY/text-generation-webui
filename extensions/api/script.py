import time

import extensions.api.blocking_api as blocking_api
import extensions.api.streaming_api as streaming_api
from extensions.api.streaming_api import DreamiaAPI
from extensions.api.tgi_inference import TGIParams
from modules import shared
from modules.logging_colors import logger


params = {
    "inference_api_endpoint": ""
}

def setup():
    logger.info("The current API is awesome, but it is deprecated by the original upstream developer. They are maintaining an OpenAI compatible API that does not support Websockets.")
    blocking_api.start_server(shared.args.api_blocking_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
    if shared.args.public_api:
        time.sleep(5)

    TGIParams.api_url = params["inference_api_endpoint"]
    DreamiaAPI.base_url = params["inference_api_endpoint"]

    streaming_api.start_server(shared.args.api_streaming_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
