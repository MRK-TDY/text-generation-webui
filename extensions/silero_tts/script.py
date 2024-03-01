import html
import json
import random
import time
from pathlib import Path

import gradio as gr
import torch
import elevenlabs

from extensions.silero_tts import tts_preprocessor
from modules import chat, shared, ui_chat
from modules.utils import gradio

from os import path
from pydub import AudioSegment

torch._C._jit_set_profiling_mode(False)


tts_modes = ['elevenlabs', 'silero', 'off']

elevenlabs_voices = [
    'Rachel',
    'Clyde',
    'Domi',
    'Dave',
    'Fin',
    'Bella',
    'Antoni',
    'Thomas',
    'Charlie',
    'Emily',
    'Elli',
    'Callum',
    'Patrick',
    'Harry',
    'Liam',
    'Dorothy',
    'Josh',
    'Arnold',
    'Charlotte',
    'Matilda',
    'Matthew',
    'James',
    'Joseph',
    'Jeremy',
    'Michael',
    'Ethan',
    'Gigi',
    'Freya',
    'Grace',
    'Daniel',
    'Serena',
    'Adam',
    'Nicole',
    'Jessie',
    'Ryan',
    'Sam',
    'Glinda',
    'Giovanni',
    'Mimi',
]

params = {
    'tts_mode': "silero", # "elevenlabs", "silero", or "off"
    'speaker': 'en_56',
    'elevenlabs_speaker': 'Adam',
    'language': 'English',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'show_text': False,
    'autoplay': True,
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
    'local_cache_path': ''  # User can override the default cache path to something other via settings.json
}

current_params = params.copy()

last_sentence_index = 0

with open(Path("extensions/silero_tts/languages.json"), encoding='utf8') as f:
    languages = json.load(f)

voice_pitches = ['x-low', 'low', 'medium', 'high', 'x-high']
voice_speeds = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})


def xmlesc(txt):
    return txt.translate(table)


def load_model():
    torch_cache_path = torch.hub.get_dir() if params['local_cache_path'] == '' else params['local_cache_path']
    model_path = torch_cache_path + "/snakers4_silero-models_master/src/silero/model/" + params['model_id'] + ".pt"
    if Path(model_path).is_file():
        print(f'\nUsing Silero TTS cached checkpoint found at {torch_cache_path}')
        model, example_text = torch.hub.load(repo_or_dir=torch_cache_path + '/snakers4_silero-models_master/', model='silero_tts', language=languages[params['language']]["lang_id"], speaker=params['model_id'], source='local', path=model_path, force_reload=True)
    else:
        print(f'\nSilero TTS cache not found at {torch_cache_path}. Attempting to download...')
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=languages[params['language']]["lang_id"], speaker=params['model_id'])
    model.to(params['device'])
    return model


def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]

    return history


def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]

    return history


def state_modifier(state):
    if not params['tts_mode'] in tts_modes or params['tts_mode'] == "off":
        return state

    # state['stream'] = False
    return state


def input_modifier(string, state):
    global last_sentence_index
    if not params['tts_mode'] in tts_modes or params['tts_mode'] == "off":
        return string

    # shared.processing_message = "*Is recording a voice message...*"
    
    last_sentence_index = 0
    return string


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def save_audio_to_file(state, string_to_voice, original_string):
    character = "character"
    if 'character_menu' in state:
        character = state['character_menu']
    output_file = Path(f'extensions/silero_tts/outputs/{character}_{int(time.time_ns())}.wav')
    prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])
    silero_input = f'<speak>{prosody}{xmlesc(string_to_voice.lower())}</prosody></speak>'
    if params['tts_mode'] == "elevenlabs":
        try:
            elevenlabs_output_file = Path(f'extensions/silero_tts/outputs/{character}_{int(time.time_ns())}.mp3')
            audio = elevenlabs.generate(text=string_to_voice, voice=params['elevenlabs_speaker'], api_key=shared.args.elevenlabs_api_key)
            elevenlabs.save(audio, str(elevenlabs_output_file)) # mp3 format
            print(f'Elevenlabs audio saved to {elevenlabs_output_file}')
            audio = AudioSegment.from_file(elevenlabs_output_file, format="mp3")
            audio.export(output_file, format="wav")
        except Exception as e:
            print("Error generating audio with Elevenlabs: ", e)
            model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))
    else:
        model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))
    # autoplay = 'autoplay' if params['autoplay'] else ''
    autoplay = ''
    string_to_voice = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
    string_to_voice += f'\n\n{original_string}'

    return string_to_voice

def output_modifier(string, state):
    global model, current_params, streaming_state, last_sentence_index

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if not params['tts_mode'] in tts_modes or params['tts_mode'] == "off":
        return string
    
    unsaid_string = string[last_sentence_index::]

    if len(unsaid_string) == 0:
        return string

    unescaped_string = string
    previous_sentence_index = last_sentence_index

    string_to_voice = tts_preprocessor.preprocess(unsaid_string)

    if string_to_voice == '':
        string_to_voice = '*Empty reply, try regenerating*'
    else:
        string_to_voice = save_audio_to_file(state, string_to_voice, unsaid_string)

    relevant_string = string_to_voice + "\n\n"
    if previous_sentence_index != 0:
        relevant_string = unescaped_string[0:previous_sentence_index] + "\n\n" + relevant_string

    final_string = relevant_string
    last_sentence_index = len(relevant_string)

    return final_string


def output_stream_modifier(string, state):
    global model, current_params, streaming_state, last_sentence_index

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if not params['tts_mode'] in tts_modes or params['tts_mode'] == "off":
        return string

    # unescaped_string = html.unescape(string)
    unescaped_string = string
    string_to_voice = ''
    previous_sentence_index = last_sentence_index

    while True:
        sentence, last_sentence_index = get_next_sentence(unescaped_string, last_sentence_index)
        if sentence is None:
            break

        string_to_voice += sentence
        print(f'string_to_voice: {string_to_voice} ({len(string_to_voice)})')

    if len(string_to_voice) <= 0:
        return string

    original_string = string_to_voice
    string_to_voice = tts_preprocessor.preprocess(string_to_voice)

    if string_to_voice == '':
        string_to_voice = '*Empty reply, try regenerating*'
    else:
        string_to_voice = save_audio_to_file(state, string_to_voice, original_string)

    relevant_string = string_to_voice + "\n\n"
    if previous_sentence_index != 0:
        relevant_string = unescaped_string[0:previous_sentence_index] + "\n\n" + relevant_string

    final_string = relevant_string + unescaped_string[last_sentence_index:-1]
    last_sentence_index = len(relevant_string)

    # print(f'string: {final_string} ({len(final_string)}); last_sentence_index: {last_sentence_index}\n')
    return final_string

def get_next_sentence(source, start = 0):
    sentence_stop = ('...', '.', '?', '!', '!!!', '!!', '??', '?!', '?!?')

    i = start
    M = max(len(s) for s in sentence_stop)
    L = len(source)

    while i <= L:
        m = M
        while m > 1:
            chunk = source[i:i + m]
            if chunk in sentence_stop:
                return source[start:i + len(chunk)], i + len(chunk)
            m -= 1
        else:
            m = 1
        i += m
    return None, start

def setup():
    global model
    model = load_model()


def random_sentence():
    with open(Path("extensions/silero_tts/harvard_sentences.txt")) as f:
        return random.choice(list(f))


def voice_preview(string):
    global model, current_params, streaming_state

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    string = tts_preprocessor.preprocess(string or random_sentence())
    string_output = save_audio_to_file({}, string, "")
    string_output = string_output.replace('controls', 'controls autoplay')

    return string_output


def say_verbatim_tts(string):
    global model, current_params, streaming_state

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    string = tts_preprocessor.preprocess(string or random_sentence())
    string_output = save_audio_to_file({}, string, "")

    return string_output


def language_change(lang):
    global params
    params.update({"language": lang, "speaker": languages[lang]["default_voice"], "model_id": languages[lang]["model_id"]})
    return gr.update(choices=languages[lang]["voices"], value=languages[lang]["default_voice"])


def custom_css():
    path_to_css = Path(__file__).parent.resolve() / 'style.css'
    return open(path_to_css, 'r').read()


def ui():
    # Gradio elements
    with gr.Accordion("Silero TTS"):
        with gr.Row():
            tts_mode = gr.Dropdown(value=params['tts_mode'], choices=tts_modes, label='TTS Mode')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        
        with gr.Row():
            elevenlabs_speaker = gr.Dropdown(value=params['elevenlabs_speaker'], choices=elevenlabs_voices, label='Elevenlabs Voice')
        
        with gr.Row():
            language = gr.Dropdown(value=params['language'], choices=sorted(languages.keys()), label='Language')
            voice = gr.Dropdown(value=params['speaker'], choices=languages[params['language']]["voices"], label='TTS voice')
        with gr.Row():
            v_pitch = gr.Dropdown(value=params['voice_pitch'], choices=voice_pitches, label='Voice pitch')
            v_speed = gr.Dropdown(value=params['voice_speed'], choices=voice_speeds, label='Voice speed')

        with gr.Row():
            preview_text = gr.Text(show_label=False, placeholder="Preview text", elem_id="silero_preview_text")
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr).then(
        remove_tts_from_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)

    # Toggle message text in history
    show_text.change(
        lambda x: params.update({"show_text": x}), show_text, None).then(
        toggle_text_in_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    # Event functions to update the parameters in the backend
    tts_mode.change(lambda x: params.update({"tts_mode": x}), tts_mode, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    language.change(language_change, language, voice, show_progress=False)
    voice.change(lambda x: params.update({"speaker": x}), voice, None)
    elevenlabs_speaker.change(lambda x: params.update({"elevenlabs_speaker": x}), elevenlabs_speaker, None)
    v_pitch.change(lambda x: params.update({"voice_pitch": x}), v_pitch, None)
    v_speed.change(lambda x: params.update({"voice_speed": x}), v_speed, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)
