from flask import Flask, request, render_template
import dill as pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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

# Tambahkan fungsi ke namespace
import __main__
__main__.preprocess_text = preprocess_text

# Memuat model
model_filename = './model/trained_model_2.pkl'
with open(model_filename, 'rb') as file:
    pipe_svc = pickle.load(file)

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
    try:
        email_text = request.form['email_text']
        email_text = preprocess_text(email_text)
        prediction = pipe_svc.predict([email_text])
        result = 'spam' if prediction[0] == 1 else 'ham'
        return render_template('form.html', prediction=result)
    except Exception as e:
        # Menangkap dan mencetak kesalahan
        print(f"Error: {e}")
        return render_template('form.html', prediction='Error processing request')

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    app.run(debug=True)