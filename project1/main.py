from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
from werkzeug.utils import secure_filename

import os


# google cloud imports
from google.cloud import speech
from google.cloud import texttospeech_v1

app = Flask(__name__)

# Configure upload folder
STT_FOLDER = 'uploads/stt'
TTS_FOLDER = 'uploads/tts'
ALLOWED_EXTENSIONS = {'wav'}
app.config['STT_FOLDER'] = STT_FOLDER
app.config['TTS_FOLDER'] = TTS_FOLDER


os.makedirs(STT_FOLDER, exist_ok=True)
os.makedirs(TTS_FOLDER, exist_ok=True)

stt_client=speech.SpeechClient()
tts_client = texttospeech_v1.TextToSpeechClient()

def recognize_speech(input_audio):
  audio=speech.RecognitionAudio(content=input_audio)

  config=speech.RecognitionConfig(
  language_code="en-US",
  model="latest_long",
  audio_channel_count=1,
  enable_word_confidence=True,
  enable_word_time_offsets=True,
  )

  operation=stt_client.long_running_recognize(config=config, audio=audio)

  response=operation.result(timeout=90)

  txt = ''
  for result in response.results:
    txt = txt + result.alternatives[0].transcript + '\n'

  return txt


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_stt_files():
    files = []
    for filename in os.listdir(STT_FOLDER):
        if allowed_file(filename):
            files.append(filename)
    files.sort(reverse=True)
    return files


def get_tts_files():
    files = []
    for filename in os.listdir(TTS_FOLDER):
        if allowed_file(filename):
            files.append(filename)
    files.sort(reverse=True)
    return files

@app.route('/')
def index():
    stt_files = get_stt_files()
    tts_files = get_tts_files()
    return render_template('index.html', stt_files=stt_files, tts_files=tts_files)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        flash('No audio data')
        return redirect(request.url)
    file = request.files['audio_data']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
        file_path = os.path.join(app.config['STT_FOLDER'], filename)
        file.save(file_path)

        f = open(file_path, 'rb')
        audio_data = f.read()
        f.close()

        text = recognize_speech(audio_data)
        txt_filename = filename + '.txt'
        txt_filepath = os.path.join(app.config['STT_FOLDER'], txt_filename)

        try:
            with open(txt_filepath, 'w') as txt_file:
                txt_file.write(text)
        except IOError as e:
            print(f"Error saving transcript: {e}")

    return redirect('/') #success

@app.route('/upload/<filename>')
def get_file(filename):
    return send_file(filename)

    
@app.route('/upload_text', methods=['POST'])
def upload_text():
    text = request.form['text']

    if not text:
        return redirect('/')

    # generate filename
    filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
    audio_path = os.path.join(TTS_FOLDER, filename)
    txt_path = os.path.join(TTS_FOLDER, f"{filename}.txt")

    # tts config
    synthesis_input = texttospeech_v1.SynthesisInput(text=text)
    voice = texttospeech_v1.VoiceSelectionParams(
        language_code="en-UK"
    )
    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16
    )

    # call client to generate
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # save audio and text
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(response.audio_content)
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text)

    return redirect('/') #success

@app.route('/script.js',methods=['GET'])
def scripts_js():
    return send_file('./script.js')

@app.route('/stt/<filename>')
def stt_file(filename):
    return send_from_directory(app.config['STT_FOLDER'], filename)

@app.route('/tts/<filename>')
def tts_file(filename):
    return send_from_directory(app.config['TTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)