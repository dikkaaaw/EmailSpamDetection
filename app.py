from flask import Flask, request, render_template
import joblib
import nltk
import os
from preprocess import preprocess_text  # Impor fungsi preprocess_text

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memastikan dataset NLTK diunduh
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Memuat model
model_filename = os.path.join(os.path.dirname(__file__), 'model', 'trained_model.pkl')
print(f"Loading model from {model_filename}")

# Definisikan preprocessing fungsi dalam konteks joblib
def custom_unpickler():
    import preprocess
    preprocess.preprocess_text = preprocess_text

pipe_svc = joblib.load(model_filename)

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
