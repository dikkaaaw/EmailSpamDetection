from flask import Flask, request, render_template
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Fungsi preprocess_text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalnum()]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Memastikan dataset NLTK diunduh
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Memuat model dengan globals
model_filename = os.path.join(os.path.dirname(__file__), 'model', 'trained_model.pkl')
print(f"Loading model from {model_filename}")
pipe_svc = joblib.load(model_filename, mmap_mode=None, globals={'preprocess_text': preprocess_text})

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
    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=False)
