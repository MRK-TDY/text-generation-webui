import asyncio
import json
import ssl
from threading import Thread
import traceback

import httpx
from websockets.server import serve
import http
import urllib.parse
from functools import reduce

import copy
import secrets

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared,
    with_api_lock
)
from extensions.api.tgi_inference import generate_chat_reply as tgi_chat_reply
from extensions.api.tgi_inference import classify_emotion
from modules import shared
from modules.chat import replace_character_names
from modules.text_generation import generate_reply
from modules.logging_colors import logger

from extensions.silero_tts import script as tts_script
from extensions.intent_detection import script as intent_script
from extensions.knowledge_management import script as km_script

import re

PATH = '/api/v1/stream'


class DreamiaAPI:
    base_url = None


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

modes = ['chat', 'chat-instruct', 'instruct', 'verbatim', 'start_interaction', 'stop_interaction', 'start_session']


async def mode_start_session(websocket, body):
    # check generate_params['player_id'] for the key to identify the player.
    # if non existent create new.
    # if key is defined but no data is available, assume new.

    player_id = body.get('player_id', '')
    if not player_id:
        player_id = secrets.token_urlsafe(16)

    await websocket.send(json.dumps({
        'event': 'stream_start_session',
        'message_num': 0,
        'player_id': player_id
    }))
    return


async def mode_start_interaction(websocket, body):
    # we might not need this, but just in case.
    await websocket.send(json.dumps({
        'event': 'stream_start_interaction',
        'message_num': 0
    }))
    return


async def mode_stop_interaction(websocket, body):
    # update the memory system with the current history.
    history = body['history']['internal']
    player_id = body.get('player_id', '')
    player = body.get("name1")
    npc = body.get("name2")

    latest_history_index = body['history'].get("latest_history_index", 0)
    relevant_history = history[latest_history_index::]
    if len(relevant_history) == 0:
        await websocket.send(json.dumps({
            'event': 'stream_stop_interaction',
            'message_num': 0
        }))
        return

    entries = []
    text = ""
    for dialogue_round in relevant_history:
        text += f"{player}: {dialogue_round[0]}\n"
        text += f"{npc}: {dialogue_round[1]}\n"

    entry = {
        "text": text,
        "payload": {
            "filter_key": f"{npc}_{player_id}"
        },
        "summarize": True
    }
    entries.append(entry)

    await km_script.add_memory(entries)

    await websocket.send(json.dumps({
        'event': 'stream_stop_interaction',
        'message_num': 0
    }))
    return


async def mode_verbatim(websocket, body):
    user_input = body['user_input']
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    _continue = body.get('_continue', False)

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

    if generate_params['mode'] == "verbatim":
        logger.info("Verbatim mode is enabled.")
        message = replace_character_names(user_input, generate_params['name1'], generate_params['name2'])
        await say_verbatim(websocket, message, generate_params)
        return


async def mode_chat_any(websocket, body):
    user_input = body['user_input']
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    regenerate = body.get('regenerate', False)
    _continue = body.get('_continue', False)

    # Check if the user input matches any of the intents
    emotion, triggered_intents = await asyncio.gather(
        classify_emotion(generate_params, user_input),
        check_intent(user_input, generate_params['player_intents']),
    )
    if len(triggered_intents) > 0:
        history = generate_params["history"]
        internal = [user_input, ""]
        history['internal'].append(internal)

        history['visible'].append(internal)
        # backward compatibility
        history['triggered_intent_id'] = triggered_intents[0]
        message_num = 0
        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'history': history,
            'triggered_intents': triggered_intents,
        }))
        message_num += 1
        await websocket.send(json.dumps({
            'event': 'stream_end',
            'message_num': message_num
        }))
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

    generate_params["context"] = generate_params["context"].replace("\r\n", "\n")
    player_id = body.get('player_id', '')
    npc = generate_params.get("name2")
    history = generate_params['history']['internal']
    history = [message for dialogue_round in history for message in dialogue_round] if len(history) > 0 else []

    world_knowledge_context, character_knowledge_context, player_knowledge_context = await asyncio.gather(
        km_script.get_context(user_input=user_input, history=history, filters=["world"], top_k=3),
        km_script.get_context(user_input=user_input, history=history, filters=[npc], top_k=3),
        km_script.get_context(user_input=user_input, history=history, filters=[f"{npc}_{player_id}"], top_k=2)
    )
    knowledge_context = world_knowledge_context + character_knowledge_context + player_knowledge_context

    generate_params["context"] = generate_params["context"].replace("<knowledge_injection>", knowledge_context)

    awareness_injection = ""
    for attr, value in generate_params["awareness"].items():
        awareness_injection += value + "\n"
    generate_params["context"] = generate_params["context"].replace("<awareness_injection>", awareness_injection)
    needs_injection = ""
    for attr, value in generate_params["wants"].items():
        needs_injection += value + "\n"
    generate_params["context"] = generate_params["context"].replace("<needs_injection>", needs_injection)
    extra_context_injection = f"Current mood: You feel {emotion}.\n"
    for attr, value in generate_params["extra_context"].items():
        extra_context_injection += value + "\n"
    generate_params["context"] = generate_params["context"].replace("<extra_context_injection>", extra_context_injection)

    # text-generation-webui reply
    # generator = generate_chat_reply(
    #     user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
    generator = tgi_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

    last_sentence_index = 0
    message_num = 0
    character_sentences = []
    new_history = []
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

            character_sentence = a['visible'][-1][1][last_sentence_index:].strip()
            character_sentence = character_sentence.replace("\n", "")
            # replace anything between <audio scr=""></audio>
            character_sentence = re.sub(r'<audio src=".*?></audio>', '', character_sentence)
            character_sentences.append(character_sentence)

            last_sentence_index = generate_params['tts_last_sentence_index']

        new_history = a
        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'history': a
        }))
        message_num += 1

    character_intents = await asyncio.gather(
        *[check_intent(character_sentence, generate_params['character_intents'], 0.7)
          for character_sentence in character_sentences]
    )
    character_intents = reduce(lambda x, y: x + y, character_intents, [])
    if len(character_intents) > 0:
        character_intents = list(set(character_intents))
        await websocket.send(json.dumps({
            'event': 'intent_triggered',
            'message_num': message_num,
            'triggered_intents': character_intents,
            'history': new_history
        }))
        message_num += 1

    response = await log_response(player_id, generate_params["context"], new_history["internal"],
                                  generate_params["name2"], generate_params["name1"])
    logger.info(response)

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


async def _handle_chat_stream_message(websocket, message):
    MODE_MAP = {
        "start_session": mode_start_session,
        "start_interaction": mode_start_interaction,
        "stop_interaction": mode_stop_interaction,
        "chat": mode_chat_any,
        "chat-instruct": mode_chat_any,
        "verbatim": mode_verbatim,
        "instruct": mode_chat_any,
    }

    try:
        body = json.loads(message)
        # logger.info(body)
        generate_params = build_parameters(body, chat=True)

        mode = generate_params.get('mode', '')
        if mode not in modes:
            mode = "chat-instruct"

        await MODE_MAP[mode](websocket, body)
        return
    except Exception as e:
        logger.error(traceback.format_exc())
        await websocket.send(json.dumps({
            "event": "stream_end",
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


async def check_intent(phrase, intents, threshold = 0.8):
    # user_input is empty
    if not phrase or not intents:
        return []

    tasks = []
    for intent in intents:
        task = intent_script.intent_similarity(phrase, intent['training_phrases'])
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    intents_scores = {intent['id']: result for intent, result in zip(intents, results)}

    for intent_id, intent_score in intents_scores.items():
        logger.info(f'Score for {intent_id}: {intent_score}')

    triggerred_intents = [intent_id for intent_id, intent_score in intents_scores.items() if intent_score > threshold]
    return triggerred_intents


async def log_response(player_id, context, history, npc_name, player_name):
    messages = []
    for player_message, npc_message in history:
        messages.append(f"{player_name}: {player_message}")
        messages.append(f"{npc_name}: {npc_message}")

    endpoint = DreamiaAPI.base_url + "/log"
    client = httpx.AsyncClient()
    response = await client.post(endpoint, json={
        "player_id": player_id,
        "context": context,
        "messages": messages
    })
    await client.aclose()
    return response
