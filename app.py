from flask import Flask, render_template, request, jsonify
import os
import whisper
from api.electra_api import electra_api

app = Flask(__name__)

app.register_blueprint(electra_api)

model = whisper.load_model("medium")  # 원하는 모델 크기로 선택

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stt', methods=['POST'])
def stt():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일 이름이 없습니다.'}), 400

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    result = model.transcribe(file_path, language='ko')  # 한국어
    text = result["text"]

    output_file_path = os.path.join('output', file.filename.replace('.mp3', '.txt'))
    os.makedirs('output', exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    os.remove(file_path)

    return jsonify({'text': text, 'output_file': output_file_path}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



