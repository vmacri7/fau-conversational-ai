from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
from werkzeug.utils import secure_filename

import os


# google cloud imports
from google.cloud import speech
from google.cloud import texttospeech_v1
from google.cloud import language_v2

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
sentiment_client = language_v2.LanguageServiceClient()


def analyze_sentiment(content):
    document_type = language_v2.Document.Type.PLAIN_TEXT

    # we will assume the user is speaking english
    language_code = "en"

    document = {
        "content": content,
        "type_": document_type,
        "language_code": language_code,
    }

    encoding_type = language_v2.EncodingType.UTF8

    response = sentiment_client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    # return sentiment score (-1 to 1)
    return response.document_sentiment.score


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
        if allowed_file(filename):  # only .wav
            sent_filename = filename + '_sentiment.txt'
            sent_filepath = os.path.join(STT_FOLDER, sent_filename)
            sentiment_score = None
            sentiment_label = None
            if os.path.exists(sent_filepath):
                # parse sentiment file
                with open(sent_filepath, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Sentiment Score:"):
                            try:
                                sentiment_score = round(float(line.split(":")[1].strip()), ndigits=2)
                            except ValueError:
                                sentiment_score = None
                        elif line.startswith("Sentiment:"):
                            sentiment_label = line.split(":")[1].strip()
            files.append({
                'filename': filename,
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label
            })
    # newest to oldest
    files.sort(key=lambda x: x['filename'], reverse=True)
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
    print(stt_files)
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

        sentiment_score = analyze_sentiment(text)
        print(f"Document sentiment score: {sentiment_score}")

        # threshold sentiment score
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # save sentiment to a file 
        sentiment_filename = filename + '_sentiment.txt'
        sentiment_filepath = os.path.join(app.config['STT_FOLDER'], sentiment_filename)
        try:
            with open(sentiment_filepath, 'w') as sentiment_file:
                sentiment_file.write(f"Sentiment Score: {sentiment_score}\nSentiment: {sentiment_label}\n")
        except IOError as e:
            print(f"Error saving sentiment analysis: {e}")

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