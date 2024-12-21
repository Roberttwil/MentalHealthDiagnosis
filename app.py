import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memeriksa status Normal atau Penyakit Mental
def check_if_normal(predictions, targets):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    if np.all(predictions == 0):
        return "Normal", []
    else:
        detected_labels = targets.columns[predictions[0] == 1].tolist()
        return f"Penyakit Mental Terdeteksi: {', '.join(detected_labels)}", detected_labels

# Fungsi untuk memberikan saran berdasarkan hasil prediksi
def get_recommendations(detected_labels):
    recommendations = {
        "Bipolar disorder": "Konsultasikan dengan psikiater untuk evaluasi lebih lanjut dan manajemen suasana hati.",
        "Schizophrenia": "Segera temui profesional kesehatan mental untuk diagnosis dan pengobatan yang tepat.",
        "Depression": "Cobalah berbicara dengan konselor atau psikolog dan pertimbangkan terapi atau dukungan kelompok.",
        "Anxiety disorder": "Latih teknik relaksasi seperti meditasi atau pernapasan dalam dan konsultasikan ke psikolog.",
        "PTSD": "Dukungan dari terapis atau kelompok pendukung dapat membantu Anda mengelola gejala PTSD.",
    }
    return [recommendations[label] for label in detected_labels]

# Load dataset dan model
file_path = 'Mentalillness.csv'
data = pd.read_csv(file_path)
model = joblib.load('mental_health_model.pkl')

# Hapus kolom yang tidak relevan
data = data.drop(columns=['Sleep disturbance.1', 'Intrusive memories or flashback'])

# Memisahkan fitur dan target
features = data.drop(columns=['ID', 'Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD'])
targets = data[['Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD']]

questions = features.columns

# Streamlit UI
st.title("Prediksi Kesehatan Mental")

st.write("Silakan jawab pertanyaan berikut dengan memilih Ya atau Tidak:")

answers = {}

# Input pengguna untuk setiap gejala
for idx, question in enumerate(questions):
    question_translated = question.replace('_', ' ').capitalize()
    question_with_number = f"{idx+1}. Do you experience {question_translated}?"
    answer = st.radio(
        question_with_number,
        options=['Belum memilih', 'Ya', 'Tidak'],
        index=0
    )
    if answer == 'Belum memilih':
        answers[question] = None
    else:
        answers[question] = 1 if answer == 'Ya' else 0

if st.button("Prediksi"):
    if None in answers.values():
        st.write("Harap isi semua pertanyaan sebelum melanjutkan!")
    else:
        user_input = np.array([list(answers.values())]).reshape(1, -1)
        prediction = model.predict(user_input)
        result, detected_labels = check_if_normal(prediction, targets)

        # Menampilkan hasil prediksi
        st.write(f"Hasil Prediksi: {result}")

        # Visualisasi jawaban pengguna
        yes_count = sum(value == 1 for value in answers.values())
        no_count = sum(value == 0 for value in answers.values())
        total = yes_count + no_count

        labels = ['Ya', 'Tidak']
        counts = [yes_count, no_count]
        percentages = [count / total * 100 for count in counts]

        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct=lambda p: f'{p:.1f}% ({int(p * total / 100)})', 
               colors=['green', 'red'], startangle=90, textprops={'fontsize': 12})
        ax.set_title('Distribusi Jawaban Anda')
        st.pyplot(fig)

        # Menampilkan jumlah dan persentase
        st.write(f"Jumlah jawaban 'Ya': {yes_count} ({percentages[0]:.1f}%)")
        st.write(f"Jumlah jawaban 'Tidak': {no_count} ({percentages[1]:.1f}%)")

        # Menampilkan saran jika ada penyakit mental yang terdeteksi
        if detected_labels:
            st.subheader("Saran untuk Anda:")
            recommendations = get_recommendations(detected_labels)
            for rec in recommendations:
                st.write(f"- {rec}")
