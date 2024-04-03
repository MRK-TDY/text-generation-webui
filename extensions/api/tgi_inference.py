from modules.extensions import apply_extensions
from modules.chat import get_stopping_strings, replace_character_names, get_generation_prompt, get_max_prompt_length, get_encoded_length
import modules.shared as shared
from modules.logging_colors import logger
from jinja2.sandbox import ImmutableSandboxedEnvironment

import requests
import html
import copy
import re
import ast
import time
from functools import partial

jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

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
    logger.info(all_stop_strings)
    pattern = re.compile(r'\*\*.*?\*\*|\([^)]*\)|\[.*?\]')
    with requests.post(f'{TGIParams.api_url}/generate-stream', json=payload, stream=True) as response:
        # Ensure the request was successful
        response.raise_for_status()
        reply = ""
        # Consume the response incrementally
        for part_reply in response.iter_content(chunk_size=None):
        # Generate
        # for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
            reply += part_reply.decode('utf-8').replace("</s>", "")
            reply = pattern.sub('', reply)
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

def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    chat_template_str = state['chat_template_str']
    if state['mode'] != 'instruct':
        chat_template_str = replace_character_names(chat_template_str, state['name1'], state['name2'])

    chat_template = jinja_env.from_string(chat_template_str)
    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_renderer = partial(chat_template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
    instruct_renderer = partial(instruction_template.render, add_generation_prompt=False)

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '':
            context = replace_character_names(state['context'], state['name1'], state['name2'])
            messages.append({"role": "system", "content": context})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    # user_input = user_input.strip()
    # if user_input and not impersonate and not _continue:
    #     messages.append({"role": "user", "content": user_input})

    def remove_extra_bos(prompt):
        for bos_token in ['<s>', '<|startoftext|>']:
            while prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]

        return prompt

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            prompt = remove_extra_bos(prompt)
            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

            outer_messages.append({"role": "user", "content": command})
            outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instruction_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            if len(suffix) > 0:
                prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        prompt = remove_extra_bos(prompt)
        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    if shared.tokenizer is not None:
        max_length = get_max_prompt_length(state)
        encoded_length = get_encoded_length(prompt)
        while len(messages) > 0 and encoded_length > max_length:

            # Remove old message, save system message
            if len(messages) > 2 and messages[0]['role'] == 'system':
                messages.pop(1)

            # Remove old message when no system message is present
            elif len(messages) > 1 and messages[0]['role'] != 'system':
                messages.pop(0)

            # Resort to truncating the user input
            else:

                user_message = messages[-1]['content']

                # Bisect the truncation point
                left, right = 0, len(user_message) - 1

                while right - left > 1:
                    mid = (left + right) // 2

                    messages[-1]['content'] = user_message[:mid]
                    prompt = make_prompt(messages)
                    encoded_length = get_encoded_length(prompt)

                    if encoded_length <= max_length:
                        left = mid
                    else:
                        right = mid

                messages[-1]['content'] = user_message[:left]
                prompt = make_prompt(messages)
                encoded_length = get_encoded_length(prompt)
                if encoded_length > max_length:
                    logger.error(f"Failed to build the chat prompt. The input is too long for the available context length.\n\nTruncation length: {state['truncation_length']}\nmax_new_tokens: {state['max_new_tokens']} (is it too high?)\nAvailable context length: {max_length}\n")
                    raise ValueError
                else:
                    logger.warning(f"The input has been truncated. Context length: {state['truncation_length']}, max_new_tokens: {state['max_new_tokens']}, available context length: {max_length}.")
                    break

            prompt = make_prompt(messages)
            encoded_length = get_encoded_length(prompt)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt
