import signal
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow import keras

# Membuat aplikasi Flask
app = Flask(__name__)
# Mengizinkan Cross-Origin Resource Sharing (CORS)
CORS(app)

# Variabel global untuk model dan tokenizer
model = None
tokenizer = None

# Mendefinisikan mapping kategori di luar fungsi classify
category_mapping = {
    0: "asesmen",
    1: "objek",
    2: "plan",
    3: "subjek"
}

# Fungsi untuk memuat model dan tokenizer dari file
def load_model_and_tokenizer():
    global model, tokenizer
    # Memuat tokenizer dari file pickle
    with open('mytokenizer_50.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Memuat model dari file
    model = keras.models.load_model('modelGRU50.h5')

# Fungsi untuk mengklasifikasikan teks yang telah diproses
def classify_text(processed_text):
    # Mengonversi teks menjadi urutan token
    sequence = tokenizer.texts_to_sequences([processed_text])
    # Memotong atau menambah urutan agar memiliki panjang yang sama
    padded_squen = pad_sequences(sequence,
                             maxlen=400,
                             padding='post',
                             truncating='post')

    # Memperoleh probabilitas keluaran dari model
    output_probs = model.predict(padded_squen)
    # Mendapatkan kelas yang diprediksi
    predicted_classes = np.argmax(output_probs, axis=1)

    return predicted_classes

# Endpoint Flask untuk mengklasifikasikan teks
@app.route('/classify', methods=['POST'])
def classify():
    # Mendapatkan data JSON dari permintaan POST
    data = request.get_json()
    input_text = data['text']
    # Memisahkan teks input menjadi bagian-bagian
    text_parts = input_text.split(". ")
    classified_results = []

    # Menyimpan hasil klasifikasi untuk setiap kategori
    category_results = {
        "asesmen": [],
        "objek": [],
        "plan": [],
        "subjek": []
    }

    # Mengklasifikasikan setiap bagian teks
    for part in text_parts:
        results = classify_text(part)
        classified_results.append(results)

    # Menggabungkan hasil klasifikasi menjadi satu array
    merged_results = np.array(classified_results).flatten()

    # Memasukkan hasil klasifikasi ke dalam kategori yang sesuai
    for category_code, part in zip(merged_results, text_parts):
        category_name = category_mapping[category_code]
        category_results[category_name].append(part)

    # Menggabungkan hasil klasifikasi untuk setiap kategori
    merged_classifications = [
        {"category": category, "text": ", ".join(parts)}
        for category, parts in category_results.items()
    ]
    
    # Mengembalikan hasil klasifikasi sebagai respons JSON
    return jsonify(merged_classifications)

# Fungsi penanganan sinyal untuk menangani interupsi (Ctrl+C)
def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

# Main block untuk menjalankan aplikasi Flask
if __name__ == '__main__':
    # Menangani sinyal interupsi (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Memuat model dan tokenizer
    load_model_and_tokenizer()
    # Menjalankan aplikasi Flask dalam mode debug
    app.run(debug=True)