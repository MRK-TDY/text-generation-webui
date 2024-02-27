import asyncio
import json
import ssl
from threading import Thread

from websockets.server import serve
import http
import urllib.parse

import copy

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared,
    with_api_lock
)
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import generate_reply
from modules.logging_colors import logger

from extensions.silero_tts import script as tts_script

import re
import zlib
import base64

PATH = '/api/v1/stream'


@with_api_lock
async def _handle_stream_message(websocket, message):
    message_copy = message
    try:
        message = zlib.decompress(message).decode('utf-8')
        message = json.loads(message)
        print(message)
    except Exception as e:
        print("Error decompressing message.")
        print(message)
        print(e)
        message = message_copy
    

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


@with_api_lock
async def _handle_chat_stream_message(websocket, message):
    
    message_copy = message
    # decompress message
    try:
        # transform message from string to bytes
    body = json.loads(message)
        message = base64.b64decode(message)
        message = zlib.decompress(message).decode('utf-8')
        body = json.loads(message)
    except Exception as e:
        print("Error decompressing message.")
        print(e)
        message = message_copy

    #body = json.loads(message)

    user_input = body['user_input']
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    regenerate = body.get('regenerate', False)
    _continue = body.get('_continue', False)

    if len(generate_params['intents']) > 0 and (generate_params['mode'] == "chat" or generate_params['mode'] == "chat-instruct"):
        # Check if the user input matches any of the intents
        await check_intent(websocket, user_input, generate_params)
        if generate_params['history']['triggered_intent_id']:
            return

    tts_script.params.update({
        "activate": generate_params['silero_tts_enable']
    })
    if generate_params['silero_tts_enable'] == True:
        print("Silero TTS is enabled.")
        tts_script.language_change(generate_params['silero_tts_language'])
        tts_script.params.update({
            "speaker": generate_params['silero_tts_speaker'],
            "voice_pitch": generate_params['silero_tts_voice_pitch'],
            "voice_speed": generate_params['silero_tts_voice_speed'],
        })
    
    if generate_params['mode'] == "verbatim":
        print("Verbatim mode is enabled.")
        await say_verbatim(websocket, user_input, generate_params)
        return

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

    message_num = 0
    for a in generator:
        for phrases in a["visible"]:
            for i, phrase in enumerate(phrases):
                phrases[i] = re.sub(r'\*.*?\*', '', phrase)
        for phrases in a["internal"]:
            for i, phrase in enumerate(phrases):
                phrases[i] = re.sub(r'\*.*?\*', '', phrase)
        try:
            json_obj = json.dumps({
                'event': 'text_stream',
                'message_num': message_num,
                'history': a
            })
            compressed_payload = zlib.compress(json_obj.encode('utf-8'))

            await websocket.send(compressed_payload)
        except Exception as e:
            print(e)

        await asyncio.sleep(0)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


async def _handle_connection(websocket, path):

    if path == '/api/v1/stream':
        async for message in websocket:
            await _handle_stream_message(websocket, message)

    elif path == '/api/v1/chat-stream':
        async for message in websocket:
            await _handle_chat_stream_message(websocket, message)

    else:
        print(f'Streaming api: unknown path: {path}')
        return


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

    sentence_tts = tts_script.say_verbatim_tts(sentence)

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
    
    # create gen params for intent questioning.
    generate_params = copy.deepcopy(state)
    generate_params['mode'] = "instruct"
    generate_params['history']['internal'] = []
    generate_params['history']['visible'] = []
    generate_params['silero_tts_enable'] = False

    tts_script.params.update({
        "activate": False
    })

    print("Checking for Intents...")

    intents_str = ""
    for intent in state['intents']:
        intents_str += "ID: {id}\n".format(id=intent['id'])
        intents_str += "Sentences:\n"
        for sentence in intent['training_phrases']:
            intents_str += "- \"{sentence}\"\n".format(sentence=sentence)
    intent_input = ("Given the following intents composed of IDs and sentences:\n"
                    "\n"
                    "{intents}\n"
                    "\n"
                    "Does the following sentence \"{input}\" belong to one of the intents? If yes then say the intent's ID. If not say  \"none\". Answer only with what was requested. Your answer is super short."
                    ).format(intents=intents_str, input=user_input)
    
    
    # print(f"{intent_input}")
    generator = generate_chat_reply(
            intent_input, generate_params, regenerate=False, _continue=False, loading_message=False)

    message_num = 0
    answer = ''
    for a in generator:
        answer = a
    
    tts_script.params.update({
        "activate": state['silero_tts_enable']
    })
    
    latest_answer = answer['internal'][-1][-1]

    # check if the answer matches any of the intents
    for intent in state['intents']:
        if intent['id'] in latest_answer:
            history['triggered_intent_id'] = intent['id']
            break

    if not history['triggered_intent_id']:
        print(f"No intent triggered. Generation: {latest_answer}")
        return

    print(f"Intent match found. Triggering {history['triggered_intent_id']}. Generation: {latest_answer}")

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