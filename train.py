import os
import tempfile
from flask import Flask, request, jsonify
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
import soundfile as sf
from pydub import AudioSegment
import speech_recognition as sr
from gtts import gTTS

app = Flask(__name__)

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/translate', methods=['POST'])
def translate():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    
    audio_file = request.files['audio']
    source_lang = request.form.get('source_lang', 'en')
    target_lang = request.form.get('target_lang', 'fr')

    # Save the uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        audio_file.save(tmp.name)
        translated_audio_path = process_audio(tmp.name, source_lang, target_lang)

    return jsonify({"translated_audio": translated_audio_path})

def process_audio(audio_path, source_lang, target_lang):
    # Transcribe audio to text
    input_text = transcribe_audio(audio_path)

    # Process input for the model
    inputs = processor(input_text, return_tensors="pt", src_lang=source_lang).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate translation
    translated = model.generate(**inputs)
    translated_text = processor.decode(translated[0], skip_special_tokens=True)

    # Convert translated text back to speech
    translated_audio_path = text_to_speech(translated_text, target_lang)
    
    return translated_audio_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    audio_path = f"translated_{lang}.mp3"
    tts.save(audio_path)
    return audio_path

if __name__ == '__main__':
    app.run(debug=True)
