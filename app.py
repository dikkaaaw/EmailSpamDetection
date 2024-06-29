from flask import Flask, request, render_template, jsonify
import joblib
import nltk
import os
from preprocess import preprocess_text  # Impor fungsi preprocess_text

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memastikan dataset NLTK diunduh
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Fungsi khusus untuk memuat model dengan globals
def custom_unpickler():
    import preprocess
    globals()['preprocess_text'] = preprocess.preprocess_text

# Memuat model
model_filename = os.path.join(os.path.dirname(__file__), 'model', 'trained_model.pkl')
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        custom_unpickler()
        pipe_svc = joblib.load(f)
else:
    raise FileNotFoundError(f"Model file {model_filename} not found. Please ensure the path is correct.")

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def form():
    return render_template('form.html')

# Route untuk memprediksi email
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    email_text = preprocess_text(email_text)
    prediction = pipe_svc.predict([email_text])
    result = 'spam' if prediction[0] == 1 else 'ham'
    response = {
        'prediction_text': f'Prediksi: {result}',
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
