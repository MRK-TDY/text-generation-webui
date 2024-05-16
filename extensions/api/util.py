import asyncio
import functools
import threading
import time
import traceback
from threading import Thread
from typing import Callable, Optional

from modules import shared
from modules.chat import load_character_memoized
from modules.chat import replace_character_names
from modules.presets import load_preset_memoized
from modules.chat import load_instruction_template

# We use a thread local to store the asyncio lock, so that each thread
# has its own lock.  This isn't strictly necessary, but it makes it
# such that if we can support multiple worker threads in the future,
# thus handling multiple requests in parallel.
api_tls = threading.local()


def build_parameters(body, chat=False):

    generate_params = {
        'max_history_len': int(body.get('max_history_len', 8)),
        'max_new_tokens': int(body.get('max_new_tokens', body.get('max_length', 250))),
        'auto_max_new_tokens': bool(body.get('auto_max_new_tokens', True)),
        'max_tokens_second': int(body.get('max_tokens_second', 0)),
        'do_sample': bool(body.get('do_sample', True)),
        'temperature': float(body.get('temperature', 1)),
        'temperature_last': bool(body.get('temperature_last', True)),
        'top_p': float(body.get('top_p', 0.8)),
        'min_p': float(body.get('min_p', 0.3)),
        'dynamic_temperature': bool(body.get('dynamic_temperature', True)),
        'dynatemp_low': float(body.get('dynatemp_low', 0.7)),
        'dynatemp_high': float(body.get('dynatemp_high', 1.5)),
        'dynatemp_exponent': float(body.get('dynatemp_exponent', 1)),
        'smoothing_factor': float(body.get('smoothing_factor', 0)),
        'smoothing_curve': float(body.get('smoothing_curve', 1)),
        'typical_p': float(body.get('typical_p', body.get('typical', 1))),
        'epsilon_cutoff': float(body.get('epsilon_cutoff', 0)),
        'eta_cutoff': float(body.get('eta_cutoff', 0)),
        'tfs': float(body.get('tfs', 1)),
        'top_a': float(body.get('top_a', 0)),
        'repetition_penalty': float(body.get('repetition_penalty', body.get('rep_pen', 1.18))),
        'presence_penalty': float(body.get('presence_penalty', body.get('presence_pen', 0))),
        'frequency_penalty': float(body.get('frequency_penalty', body.get('frequency_pen', 0))),
        'repetition_penalty_range': int(body.get('repetition_penalty_range', 1024)),
        'encoder_repetition_penalty': float(body.get('encoder_repetition_penalty', 1.0)),
        'top_k': int(body.get('top_k', 40)),
        'min_length': int(body.get('min_length', 0)),
        'no_repeat_ngram_size': int(body.get('no_repeat_ngram_size', 0)),
        'num_beams': int(body.get('num_beams', 1)),
        'penalty_alpha': float(body.get('penalty_alpha', 0)),
        'length_penalty': float(body.get('length_penalty', 1)),
        'early_stopping': bool(body.get('early_stopping', False)),
        'mirostat_mode': int(body.get('mirostat_mode', 2)),
        'mirostat_tau': float(body.get('mirostat_tau', 8)),
        'mirostat_eta': float(body.get('mirostat_eta', 0.1)),
        'grammar_string': str(body.get('grammar_string', '')),
        'guidance_scale': float(body.get('guidance_scale', 1)),
        'negative_prompt': str(body.get('negative_prompt', '')),
        'seed': int(body.get('seed', -1)),
        'add_bos_token': bool(body.get('add_bos_token', True)),
        'truncation_length': int(body.get('truncation_length', body.get('max_context_length', 2048))),
        'custom_token_bans': str(body.get('custom_token_bans', '')),
        'sampler_priority': body.get('sampler_priority', None),
        'ban_eos_token': bool(body.get('ban_eos_token', False)),
        'skip_special_tokens': bool(body.get('skip_special_tokens', True)),
        'custom_stopping_strings': '',  # leave this blank
        'stopping_strings': body.get('stopping_strings', []),
        'custom_system_message': str(body.get('custom_system_message', 'Answer the request using very short responses. Use only up to three short sentences.')),
        'prompt_lookup_num_tokens': int(body.get('prompt_lookup_num_tokens', 0)),

        'tts_mode': str(body.get('tts_mode', 'silero')),
        'silero_tts_speaker': str(body.get('silero_tts_speaker', 'en_56')),
        'silero_tts_language': str(body.get('silero_tts_language', 'English')),
        'silero_tts_voice_pitch': str(body.get('silero_tts_voice_pitch', 'medium')),
        'silero_tts_voice_speed': str(body.get('silero_tts_voice_speed', 'medium')),
        'elevenlabs_speaker': str(body.get('elevenlabs_speaker', 'Alice')),
    

        'intents': body.get('intents', []),
        'player_intents': body.get('player_intents', []),
        'character_intents': body.get('character_intents', []),

        'awareness': body.get('awareness', {}),
        'wants': body.get('wants', {}),
        'extra_context': body.get('extra_context', {}),

        'memory_fragments': body.get('memory_fragments', []),
    }

    preset_name = body.get('preset', 'None')
    if preset_name not in ['None', None, '']:
        preset = load_preset_memoized(preset_name)
        generate_params.update(preset)

    if chat:
        character = body.get('character')
        instruction_template = body.get('instruction_template', shared.default_settings['instruction_template'])
        if str(instruction_template) == "None":
            instruction_template = "Llama-v2"
        instruction_template_str = load_instruction_template(instruction_template)
        
        if str(character) == "None":
            character = "Assistant"

        name1, name2, _, greeting, context = load_character_memoized(character, str(body.get('your_name', shared.default_settings['name1'])), str(body.get('character', '')))
        name1 = str(body.get('name1', name1))
        name2 = str(body.get('name2', name2))
        context = str(body.get('context', context))
        generate_params.update({
            'mode': str(body.get('mode', 'chat')),
            'name1': name1,
            'name2': name2,
            'context': context,
            'greeting': str(body.get('greeting', greeting)),
            'turn_template': str(body.get('turn_template', '<|user|><|user-message|><|bot|><|bot-message|>')),
            'chat-instruct_command': str(body.get('chat_instruct_command', body.get('chat-instruct_command', shared.default_settings['chat-instruct_command']))),
            'history': body.get('history', {'internal': [], 'visible': []}),
            'instruction_template_str': instruction_template_str,
            'chat_template_str': str(shared.default_settings['chat_template_str']),
        })

        if body.get('mode', 'chat') == "instruct":
            context_instruct = replace_character_names(context, name1, name2)
            generate_params.update({
                'custom_system_message': "Given this context:\n\n" + context_instruct + "\n\n" + generate_params['custom_system_message'],
            })

    return generate_params


def try_start_cloudflared(port: int, tunnel_id: str, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    Thread(target=_start_cloudflared, args=[
           port, tunnel_id, max_attempts, on_start], daemon=True).start()


def _start_cloudflared(port: int, tunnel_id: str, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    try:
        from flask_cloudflared import _run_cloudflared
    except ImportError:
        print('You should install flask_cloudflared manually')
        raise Exception(
            'flask_cloudflared not installed. Make sure you installed the requirements.txt for this extension.')

    for _ in range(max_attempts):
        try:
            if tunnel_id is not None:
                public_url = _run_cloudflared(port, port + 1, tunnel_id=tunnel_id)
            else:
                public_url = _run_cloudflared(port, port + 1)

            if on_start:
                on_start(public_url)

            return
        except Exception:
            traceback.print_exc()
            time.sleep(3)

        raise Exception('Could not start cloudflared.')


def _get_api_lock(tls) -> asyncio.Lock:
    """
    The streaming and blocking API implementations each run on their own
    thread, and multiplex requests using asyncio.  If multiple outstanding
    requests are received at once, we will try to acquire the shared lock
    shared.generation_lock multiple times in succession in the same thread,
    which will cause a deadlock.

    To avoid this, we use this wrapper function to block on an asyncio
    lock, and then try and grab the shared lock only while holding
    the asyncio lock.
    """
    if not hasattr(tls, "asyncio_lock"):
        tls.asyncio_lock = asyncio.Lock()

    return tls.asyncio_lock


def with_api_lock(func):
    """
    This decorator should be added to all streaming API methods which
    require access to the shared.generation_lock.  It ensures that the
    tls.asyncio_lock is acquired before the method is called, and
    released afterwards.
    """
    @functools.wraps(func)
    async def api_wrapper(*args, **kwargs):
        async with _get_api_lock(api_tls):
            return await func(*args, **kwargs)
    return api_wrapper
