from flask import Flask, request, render_template, jsonify
import joblib
import nltk
import os
from preprocess import preprocess_text  # Pastikan ini terimpor

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
    custom_unpickler()  # Panggil fungsi untuk memastikan preprocess_text tersedia
    pipe_svc = joblib.load(model_filename)
else:
    raise FileNotFoundError(f"Model file {model_filename} not found. Please ensure the path is correct.")

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
