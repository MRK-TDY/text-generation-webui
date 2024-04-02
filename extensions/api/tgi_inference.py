from modules.extensions import apply_extensions
from modules.chat import get_stopping_strings, generate_chat_prompt
import modules.shared as shared
from modules.logging_colors import logger
import requests

import html
import copy
import re
import ast
import time


class TGIParams:
    api_url = None


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message, for_ui=for_ui):
        yield history


def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Apply extensions
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state, is_chat=True)

        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])

        # *Is typing...*
        if loading_message:
            yield {
                'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
                'internal': output['internal']
            }
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, shared.processing_message]],
                    'internal': output['internal'][:-1] + [[text, '']]
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal']
                }
    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output,
    }
    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    reply = None
    previous_reply = None
    cropped_reply = None
    visible_reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True)):

        if previous_reply is None:
            visible_reply = reply
            if state['mode'] in ['chat', 'chat-instruct']:
                visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
            visible_reply = html.escape(visible_reply)
        else:
            cropped_reply = reply.replace(previous_reply, "")

            visible_reply = visible_reply + cropped_reply

        previous_reply = reply

        if shared.stop_everything:
            output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
            if is_stream:
                output['visible'][-1][1] = apply_extensions('output-stream', output['visible'][-1][1], state, is_chat=True)
                yield output
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            if is_stream:
                visible_reply = apply_extensions('output-stream', output['visible'][-1][1], state, is_chat=True)
                output['visible'][-1][1] = visible_reply.lstrip(' ')
                yield output

    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output


def generate_reply(*args, **kwargs):
    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):
    if shared.args.verbose:
        logger.info("PROMPT=")
        print(question)
        print()

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions('state', state)
        question = apply_extensions('input', question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    last_update = -1
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    payload = {
        "prompt": question,
        "max_new_tokens": state['max_new_tokens'],
        "top_p": state['top_p'],
        "top_k": state['top_k'],
        "repetition_penalty": state['repetition_penalty'],
        "stop_sequences": all_stop_strings,
    }
    with requests.post(f'{TGIParams.api_url}/generate-stream', json=payload, stream=True) as response:
        # Ensure the request was successful
        response.raise_for_status()
        reply = ""
        # Consume the response incrementally
        for part_reply in response.iter_content(chunk_size=None):
        # Generate
        # for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
            reply += part_reply.decode('utf-8').replace("</s>", "")
            logger.info(reply)
            reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
            if escape_html:
                reply = html.escape(reply)
            if is_stream:
                cur_time = time.time()

                # Maximum number of tokens/second
                if state['max_tokens_second'] > 0:
                    diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                    if diff > 0:
                        time.sleep(diff)

                    last_update = time.time()
                    yield reply

                # Limit updates to avoid lag in the Gradio UI
                # API updates are not limited
                else:
                    if cur_time - last_update > min_update_interval:
                        last_update = cur_time
                        yield reply

            if stop_found or (state['max_tokens_second'] > 0 and shared.stop_everything):
                break

    if not is_chat:
        reply = apply_extensions('output', reply, state)

    yield reply


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found