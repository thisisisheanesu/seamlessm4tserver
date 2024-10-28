pip install -r requirements.txt
python main.py

# Example
curl -X POST -F "audio=@path_to_your_audio_file.wav" -F "source_lang=en" -F "target_lang=fr" http://127.0.0.1:5000/translate

