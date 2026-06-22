from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy predict function — replace with your real one
def predict_image(filepath):
    # Your existing prediction logic goes here
    # For demo, just a random choice:
    import random
    raw_pred = random.choice(['AI-generated', 'Real Image'])
    return raw_pred

def normalize_prediction(raw_pred):
    raw_pred = raw_pred.lower()
    if 'ai' in raw_pred:
        return "AI"
    elif 'real' in raw_pred:
        return "Real"
    else:
        return "Unknown"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    raw_prediction = predict_image(filepath)
    print(f"Raw prediction: {raw_prediction}")
    
    result = normalize_prediction(raw_prediction)
    print(f"Normalized result: {result}")
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)