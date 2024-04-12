import asyncio
import json
import ssl
from threading import Thread
import traceback
from websockets.server import serve
import http
import urllib.parse
from unidecode import unidecode

import copy

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared,
    with_api_lock
)
from extensions.api.tgi_inference import generate_chat_reply as tgi_chat_reply
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import generate_reply
from modules.logging_colors import logger

from extensions.silero_tts import script as tts_script
from extensions.intent_detection import script as intent_script
from extensions.knowledge_management import script as km_script

import re

PATH = '/api/v1/stream'


@with_api_lock
async def _handle_stream_message(websocket, message):
    message = json.loads(message)

    prompt = message['prompt']
    generate_params = build_parameters(message)
    stopping_strings = generate_params.pop('stopping_strings')
    generate_params['stream'] = True

    generator = generate_reply(
        prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

    # As we stream, only send the new bytes.
    skip_index = 0
    message_num = 0

    for a in generator:
        to_send = a[skip_index:]
        if to_send is None or chr(0xfffd) in to_send:  # partial unicode character, don't send it yet.
            continue

        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'text': to_send
        }))

        await asyncio.sleep(0)
        skip_index += len(to_send)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


async def _handle_chat_stream_message(websocket, message):
    body = json.loads(message)

    # logger.info(body)

    user_input = body['user_input']
    user_input = unidecode(user_input)
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    regenerate = body.get('regenerate', False)
    _continue = body.get('_continue', False)

    if len(generate_params['intents']) > 0 and (generate_params['mode'] == "chat" or generate_params['mode'] == "chat-instruct"):
        # Check if the user input matches any of the intents
        await check_intent(websocket, user_input, generate_params)
        if generate_params['history']['triggered_intent_id']:
            return

    do_sentence_check = False
    tts_script.params.update({
        "tts_mode": generate_params['tts_mode']
    })
    if generate_params['tts_mode'] in tts_script.tts_modes and generate_params['tts_mode'] != 'off':
        if generate_params['tts_mode'] == 'silero':
            print("Silero TTS is enabled.")
        elif generate_params['tts_mode'] == 'elevenlabs':
            print("Elevenlabs TTS is enabled.")
        tts_script.language_change(generate_params['silero_tts_language'])
        tts_script.params.update({
            "speaker": generate_params['silero_tts_speaker'],
            "voice_pitch": generate_params['silero_tts_voice_pitch'],
            "voice_speed": generate_params['silero_tts_voice_speed'],
            "elevenlabs_speaker": generate_params['elevenlabs_speaker'],
        })
        do_sentence_check = True

    if generate_params['mode'] == "verbatim":
        logger.info("Verbatim mode is enabled.")
        await say_verbatim(websocket, user_input, generate_params)
        return

    generate_params["context"] = generate_params["context"].replace("\r\n", "\n")
    history = generate_params['history']['internal']
    history = [message for dialogue_round in history for message in dialogue_round] if len(history) > 0 else []
    knowledge_context = km_script.get_context(user_input=user_input, history=history,
                                              filters=["world", generate_params["name2"]])
    generate_params["context"] = generate_params["context"].replace("<knowledge_injection>", knowledge_context)
    full_internal_history = copy.deepcopy(generate_params['history']['internal'])
    full_visible_history = copy.deepcopy(generate_params['history']['visible'])
    max_history_len = generate_params['max_history_len']
    if len(full_internal_history) > max_history_len:
        generate_params['history']['internal'] = full_internal_history[-max_history_len:]
        generate_params['history']['visible'] = full_visible_history[-max_history_len:]
    # text-generation-webui reply
    # generator = generate_chat_reply(
    #     user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
    generator = tgi_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
    generate_params['history']['internal'] = full_internal_history
    generate_params['history']['visible'] = full_visible_history

    last_sentence_index = 0
    message_num = 0
    try:
        async for a in generator:
            for phrases in a["visible"]:
                for i, phrase in enumerate(phrases):
                    phrases[i] = re.sub(r'\*.*?\*', '', phrase)
            for phrases in a["internal"]:
                for i, phrase in enumerate(phrases):
                    phrases[i] = re.sub(r'\*.*?\*', '', phrase)

            if 'tts_last_sentence_index' in generate_params:
                # if last visible phrase is empty, skip
                if do_sentence_check and \
                    last_sentence_index == generate_params['tts_last_sentence_index']:
                    await asyncio.sleep(0)
                    continue

                last_sentence_index = generate_params['tts_last_sentence_index']

            await websocket.send(json.dumps({
                'event': 'text_stream',
                'message_num': message_num,
                'history': a
            }))

            message_num += 1

        await websocket.send(json.dumps({
            'event': 'stream_end',
            'message_num': message_num
        }))

    except Exception as e:
        await websocket.send(json.dumps({
            "event": "stream_end",
            'message_num': message_num,
            "error_message": str(e)
        }))



async def _handle_connection(websocket, path):
    tasks = []
    if path == '/api/v1/stream':
        async for message in websocket:
            # Create a new task for each message and append it to the tasks list
            task = asyncio.create_task(_handle_stream_message(websocket, message))
            tasks.append(task)

    elif path == '/api/v1/chat-stream':
        async for message in websocket:
            # Create a new task for each message and append it to the tasks list
            task = asyncio.create_task(_handle_chat_stream_message(websocket, message))
            tasks.append(task)

    else:
        print(f'Streaming api: unknown path: {path}')
        return

    await asyncio.gather(*tasks)



def get_query_param(path, key):
    query = urllib.parse.urlparse(path).query
    params = urllib.parse.parse_qs(query)
    values = params.get(key, [])
    if len(values) == 1:
        return values[0]

def is_key_valid(key):
    return bool(key == shared.args.api_key)

async def _handle_process_request(path, request_headers):
    if not shared.args.api_key:
        return
    
    key = request_headers.get("Authorization", None)
    if key is None:
        return http.HTTPStatus.UNAUTHORIZED, [], b"Missing token\n"

    if not is_key_valid(key):
        return http.HTTPStatus.UNAUTHORIZED, [], b"Invalid token\n"

async def _run(host: str, port: int):
    ssl_certfile = shared.args.ssl_certfile
    ssl_keyfile = shared.args.ssl_keyfile
    ssl_verify = True if (ssl_keyfile and ssl_certfile) else False
    if ssl_verify:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(ssl_certfile, ssl_keyfile)
    else:
        context = None

    async with serve(_handle_connection, host, port, ping_interval=None, ssl=context, process_request=_handle_process_request):
        await asyncio.Future()  # Run the server forever


def _run_server(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'
    ssl_certfile = shared.args.ssl_certfile
    ssl_keyfile = shared.args.ssl_keyfile
    ssl_verify = True if (ssl_keyfile and ssl_certfile) else False

    def on_start(public_url: str):
        public_url = public_url.replace('https://', 'wss://')
        logger.info(f'Streaming API URL: \n\n{public_url}{PATH}\n')

    if share:
        try:
            try_start_cloudflared(port, tunnel_id, max_attempts=3, on_start=on_start)
        except Exception as e:
            print(e)
    else:
        if ssl_verify:
            logger.info(f'Streaming API URL: \n\nwss://{address}:{port}{PATH}\n')
        else:
            logger.info(f'Streaming API URL: \n\nws://{address}:{port}{PATH}\n')

    asyncio.run(_run(host=address, port=port))


def start_server(port: int, share: bool = False, tunnel_id=str):
    Thread(target=_run_server, args=[port, share, tunnel_id], daemon=True).start()


async def say_verbatim(websocket, user_input, state):

    # user_input is empty
    sentence = user_input
    if not sentence:
        sentence = "No input."
    history = state['history']

    sentence_tts = await tts_script.say_verbatim_tts(sentence)

    internal = [ "", sentence ]
    visible = [ "", sentence_tts ]

    history['internal'].append(internal)
    history['visible'].append(visible)

    message_num = 0
    await websocket.send(json.dumps({
        'event': 'text_stream',
        'message_num': message_num,
        'history': history
    }))

    await asyncio.sleep(0)
    message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))

async def check_intent(websocket, user_input, state):
    history = state['history']
    history['triggered_intent_id'] = ""

    # user_input is empty
    if not user_input:
        return


    print("Checking for Intents...")

    intents = {}
    for intent in state['intents']:
        intents[intent['id']] = intent_script.intent_similarity(user_input, intent['training_phrases'])
    max_intent = None
    for intent_id, intent_score in intents.items():
        if max_intent is None or intent_score > intents[max_intent]:
            max_intent = intent_id

    # tts_script.params.update({
    #     "tts_mode": "off"
    # })
    #
    # intents_str = ""
    # for intent in state['intents']:
    #     intents_str += "ID: {id}\n".format(id=intent['id'])
    #     intents_str += "Sentences:\n"
    #     for sentence in intent['training_phrases']:
    #         intents_str += "- \"{sentence}\"\n".format(sentence=sentence)
    # intent_input = ("Given the following intents composed of IDs and sentences:\n"
    #                 "\n"
    #                 "{intents}\n"
    #                 "\n"
    #                 "To which intent is the sentence \"{input}\" more related?\n"
    #                 "\n"
    #                 "Answer only with the ID or \"none\". No explanation. No other words. Super short response."
    #                 ).format(intents=intents_str, input=user_input)
    #
    # # Add instruction tags
    # intent_input = f"[INST]{intent_input}[/INST]"
    #
    # # create gen params for intent questioning.
    # generate_params = build_parameters({})
    # generate_params.update({
    #     "mode": "instruct",
    #     "tts_mode": "off",
    #     "stream": False,
    #     "max_new_tokens": 100,
    #     "temperature": 0.1,
    #     "top_p": 0,
    #     "min_p": 0,
    #     "top_k": 0,
    #     "repetition_penalty": 1.5,
    #     "presence_penalty": 2,
    #     "do_sample": False,
    #     # "num_beams": 3,
    #     # "length_penalty": -5,
    #     # "early_stopping": True,
    #     "guidance_scale": 1.5,
    #     "negative_prompt": "Long answer. Explain. Based on the given intents. Here's my answer. Sure!",
    # })
    # stopping_strings = generate_params.pop('stopping_strings')
    # generator = generate_reply(
    #     intent_input, generate_params, stopping_strings=stopping_strings, is_chat=False)
    #
    # message_num = 0
    # answer = ''
    # for a in generator:
    #     answer = a
    #
    # tts_script.params.update({
    #     "tts_mode": state['tts_mode']
    # })
    #
    # # latest_answer = answer['internal'][-1][-1]
    # latest_answer = answer
    #
    # def format_intent_id(input: str) -> str:
    #     result = input.lower()
    #     result = result.replace("_", "")
    #     result = result.strip()
    #     return result
    #
    # # check if the answer matches any of the intents
    # for intent in state['intents']:
    #     if format_intent_id(intent['id']) in format_intent_id(latest_answer):
    #         history['triggered_intent_id'] = intent['id']
    #         break
    #
    # if not history['triggered_intent_id']:
    #     print(f"No intent triggered. Generation: {latest_answer}")
    #     return
    #
    # print(f"Intent match found. Triggering {history['triggered_intent_id']}. Generation: {latest_answer}")

    if intents[max_intent] < 0.8:
        print("No intent triggered.")
        return

    history['triggered_intent_id'] = max_intent
    # fix history
    internal = [ user_input, "" ]
    history['internal'].append(internal)
    history['visible'].append(internal)

    message_num = 0
    await websocket.send(json.dumps({
        'event': 'text_stream',
        'message_num': message_num,
        'history': history
    }))

    await asyncio.sleep(0)
    message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))