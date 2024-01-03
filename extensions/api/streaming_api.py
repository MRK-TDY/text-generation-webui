import asyncio
import json
import ssl
from threading import Thread

from websockets.server import serve
import http
import urllib.parse

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


@with_api_lock
async def _handle_chat_stream_message(websocket, message):
    body = json.loads(message)

    user_input = body['user_input']
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    regenerate = body.get('regenerate', False)
    _continue = body.get('_continue', False)

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
        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'history': a
        }))

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
    if sentence == "":
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