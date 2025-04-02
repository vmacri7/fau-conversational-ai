import os
import re 
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory

# --- Vertex AI Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part

app = Flask(__name__)

# --- Configuration ---
STT_FOLDER = 'uploads/stt' 
ALLOWED_EXTENSIONS = {'wav'}
app.config['STT_FOLDER'] = STT_FOLDER

# --- Vertex AI Configuration ---
PROJECT_ID = 'conversational-ai-448621'
LOCATION = 'us-east1' 
MODEL_NAME = 'gemini-1.5-flash-001'

# --- Initialization ---
os.makedirs(STT_FOLDER, exist_ok=True)

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    print(f"Vertex AI initialized successfully. Project: {PROJECT_ID}, Location: {LOCATION}, Model: {MODEL_NAME}")
except Exception as e:
    print(f"FATAL: Could not initialize Vertex AI: {e}")
    exit()

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_llm_response(llm_text):
    """
    Parses the LLM response to extract transcript, sentiment label, and score.
    Expects format with each item starting on a new line:
    Text: [TRANSCRIPT]
    Sentiment Label: [positive|neutral|negative]
    Sentiment Score: [SCORE as float between -1 and 1]
    """
    transcript = "Error: Could not parse transcript."
    sentiment_label = "neutral" # Default
    sentiment_score = 0.0      # Default

    # Pre-clean the response text (remove potential ``` wrapper)
    llm_text = llm_text.strip().strip('```').strip()

    try:
        # Match "Text:" at the start of a line (ignoring leading whitespace), capture the rest of THAT line.
        text_match = re.search(r"^\s*Text:(.*)", llm_text, re.IGNORECASE | re.MULTILINE)

        # Match "Sentiment Label:" at the start of a line, capture the label, ensure nothing else on the line (except whitespace).
        label_match = re.search(r"^\s*Sentiment Label:\s*(positive|neutral|negative)\s*$", llm_text, re.IGNORECASE | re.MULTILINE)

        # Match "Sentiment Score:" at the start of a line, capture the score, ensure nothing else on the line (except whitespace).
        score_match = re.search(r"^\s*Sentiment Score:\s*(-?\d+(\.\d+)?)\s*$", llm_text, re.IGNORECASE | re.MULTILINE)

        if text_match:
            # Strip leading/trailing whitespace from the captured group
            transcript = text_match.group(1).strip()
            # Remove potential leading/trailing quote marks sometimes added by LLMs
            transcript = transcript.strip('"').strip("'")
            # Handle potential case where only "Text:" exists with nothing after it
            if not transcript:
                transcript = "[Empty Transcript Received]"
        else:
             print("Warning: 'Text:' field not found in LLM response.")
             # Keep default error message in transcript


        if label_match:
            sentiment_label = label_match.group(1).lower() # Ensure lowercase
        else:
             # Don't print warning if text_match also failed (likely malformed response)
             if text_match:
                 print("Warning: 'Sentiment Label:' field not found or invalid in LLM response.")


        if score_match:
            try:
                # Extract the score and convert to float
                raw_score = float(score_match.group(1))
                # Bound the score between -1.0 and 1.0
                sentiment_score = max(-1.0, min(1.0, raw_score))
            except ValueError:
                 if text_match: # Avoid warning if response is totally malformed
                    print(f"Warning: Could not parse sentiment score from '{score_match.group(1)}'. Using default 0.0.")
                    sentiment_score = 0.0 # Default if conversion fails
        else:
             if text_match: # Avoid warning if response is totally malformed
                print("Warning: 'Sentiment Score:' field not found or invalid in LLM response.")


    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw Response:\n---\n{llm_text}\n---")
        # Return defaults but potentially keep the raw text for debugging if parsing fails early
        transcript = f"Error parsing response: {e}" # Overwrite transcript with error


    # Map parsed label string to standardized label for consistency
    if sentiment_label == "positive":
        final_label = "Positive"
    elif sentiment_label == "negative":
        final_label = "Negative"
    else: # Includes 'neutral' or parsing errors/missing field
        final_label = "Neutral"

    return transcript, final_label, sentiment_score


def process_audio_with_llm(audio_bytes):
    """
    Sends audio to Vertex AI Gemini model for transcription and sentiment analysis.
    Returns transcript, sentiment label, and sentiment score.
    """
    if not model:
        print("Error: Vertex AI Model not initialized.")
        return "Error: Model not available", "Neutral", 0.0

    print("Sending audio to LLM...")
    try:
        # Create the prompt
        prompt = """Please provide an exact transcript for the audio, followed by sentiment analysis including a label and a numerical score.

Your response MUST follow this exact format, with each item on a new line:

Text: [USERS SPEECH TRANSCRIPTION HERE]
Sentiment Label: [positive|neutral|negative]
Sentiment Score: [SENTIMENT SCORE AS A FLOAT BETWEEN -1.0 AND 1.0 HERE]
"""
        # Prepare the audio part
        audio_file = Part.from_data(data=audio_bytes, mime_type="audio/wav")

        # Prepare the request contents
        contents = [audio_file, prompt]

        # Generate content
        response = model.generate_content(contents)

        print("LLM Response Received.")
        # print(f"Raw LLM Response Text:\n{response.text}") # Optional: for debugging

        # Parse the response
        return parse_llm_response(response.text)

    except Exception as e:
        print(f"An unexpected error occurred during LLM processing: {e}")
        return f"Error: Unexpected processing error ({type(e).__name__})", "Neutral", 0.0


def get_stt_files():
    """Gets list of processed audio files and their sentiment data."""
    files = []
    if not os.path.exists(STT_FOLDER):
        return files

    for filename in os.listdir(STT_FOLDER):
        # Process only the original audio files, not the derived .txt files
        if allowed_file(filename):
            base_filename, ext = os.path.splitext(filename)
            sent_filename = base_filename + '_sentiment.txt' # Adjusted naming convention
            sent_filepath = os.path.join(STT_FOLDER, sent_filename)

            sentiment_score = None
            sentiment_label = None

            if os.path.exists(sent_filepath):
                try:
                    with open(sent_filepath, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line.startswith("Sentiment Score:"):
                                try:
                                    # Extract score and round for display
                                    score_str = line.split(":", 1)[1].strip()
                                    sentiment_score = round(float(score_str), ndigits=2)
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not parse score from line in {sent_filename}: {line}")
                                    sentiment_score = 0.0 # Default on parsing error
                            elif line.startswith("Sentiment:"):
                                try:
                                    sentiment_label = line.split(":", 1)[1].strip()
                                except IndexError:
                                     print(f"Warning: Could not parse label from line in {sent_filename}: {line}")
                                     sentiment_label = "Neutral" # Default on parsing error
                except IOError as e:
                    print(f"Error reading sentiment file {sent_filepath}: {e}")
                    sentiment_label = "Error Reading"
                    sentiment_score = 0.0

            # Ensure score is float or None for template logic
            if sentiment_score is not None:
                sentiment_score = float(sentiment_score)

            files.append({
                'filename': filename, # The original .wav filename
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label
            })

    # Sort by filename (which includes timestamp), newest first
    files.sort(key=lambda x: x['filename'], reverse=True)
    return files

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page with the list of processed audio files."""
    stt_files = get_stt_files()
    # TTS files are removed
    return render_template('index.html', stt_files=stt_files)

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handles audio upload, processing via LLM, and saving results."""
    if 'audio_data' not in request.files:
        #flash('No audio data part in the request.')
        return redirect(url_for('index'))

    file = request.files['audio_data']

    if file.filename == '':
        #flash('No selected file.')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename): # Check if file exists and has allowed extension
        # Generate a secure filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_base = f"audio_{timestamp}"
        audio_filename = f"{filename_base}.wav"
        audio_filepath = os.path.join(app.config['STT_FOLDER'], audio_filename)

        try:
            # Save the uploaded audio file
            file.save(audio_filepath)
            print(f"Audio file saved to: {audio_filepath}")

            # Read the audio data for processing
            with open(audio_filepath, 'rb') as f_audio:
                audio_data = f_audio.read()

            # Process with LLM
            transcript, sentiment_label, sentiment_score = process_audio_with_llm(audio_data)

            # Define paths for output files
            txt_filename = f"{filename_base}.txt"
            sentiment_filename = f"{filename_base}_sentiment.txt" # Use base name
            txt_filepath = os.path.join(app.config['STT_FOLDER'], txt_filename)
            sentiment_filepath = os.path.join(app.config['STT_FOLDER'], sentiment_filename)

            # Save the transcript
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(transcript)
                print(f"Transcript saved to: {txt_filepath}")
            except IOError as e:
                print(f"Error saving transcript: {e}")
                #flash(f'Error saving transcript for {secure_audio_filename}.')
                # Continue to save sentiment if possible

            # Save the sentiment analysis results
            try:
                with open(sentiment_filepath, 'w', encoding='utf-8') as sentiment_file:
                    sentiment_file.write(f"Sentiment Score: {sentiment_score:.4f}\n") # Save with more precision if needed
                    sentiment_file.write(f"Sentiment: {sentiment_label}\n")
                print(f"Sentiment saved to: {sentiment_filepath}")
            except IOError as e:
                print(f"Error saving sentiment analysis: {e}")
                #flash(f'Error saving sentiment for {secure_audio_filename}.')

            #flash(f'File {secure_audio_filename} processed successfully.')

        except Exception as e:
            print(f"Error during file upload or processing: {e}")
            #flash('An error occurred during processing.')

        #flash('Invalid file type. Only .wav files are allowed.')

    return redirect(url_for('index')) # Redirect back to the main page

# Removed /upload_text route

@app.route('/stt/<filename>')
def stt_file(filename):
    """Serves files (audio, transcript, sentiment) from the STT upload folder."""
    # Basic security check: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    return send_from_directory(app.config['STT_FOLDER'], filename)


if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network if needed
    app.run(debug=True, host='0.0.0.0', port=5000)