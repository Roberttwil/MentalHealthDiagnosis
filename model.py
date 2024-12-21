import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib  # Untuk menyimpan model

# Load dataset
file_path = '/kaggle/input/mental-illness-dataset/Mentalillness.csv'  # Ganti dengan path yang benar
data = pd.read_csv(file_path)

# Hapus kolom yang tidak relevan
data = data.drop(columns=['Sleep disturbance.1', 'Intrusive memories or flashback'])  # Ganti sesuai kolom yang tidak relevan

# Memisahkan fitur dan target
features = data.drop(columns=['ID', 'Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD'])
targets = data[['Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD']]

# Membagi data menjadi training, validasi, dan test set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Menangani ketidakseimbangan kelas dengan menghitung scale_pos_weight
num_pos = sum(y_train.sum(axis=0))  # Total kelas positif
num_neg = len(y_train) * y_train.shape[1] - num_pos  # Total kelas negatif
scale_pos_weight = num_neg / num_pos
print(f"Scale Pos Weight: {scale_pos_weight}")

# Model XGBoost untuk multi-label classification
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight)

# Melatih model
model.fit(X_train, y_train)

# Memprediksi hasil pada data test
y_pred = model.predict(X_test)

# Evaluasi akurasi untuk setiap label (penyakit mental)
for idx, label in enumerate(targets.columns):
    accuracy = accuracy_score(y_test[label], y_pred[:, idx])
    print(f"Akurasi untuk {label}: {accuracy * 100:.2f}%")

# Fungsi untuk memeriksa status Normal atau Penyakit Mental
def check_if_normal(predictions):
    # Jika semua kolom target prediksi adalah 0 (tidak ada penyakit terdeteksi), maka "normal"
    if np.all(predictions == 0):
        return "Normal"
    else:
        # Menampilkan label penyakit mental yang terdeteksi
        detected_labels = targets.columns[predictions == 1].tolist()
        return f"Penyakit Mental Terdeteksi: {', '.join(detected_labels)}"

# Menerapkan fungsi ke hasil prediksi untuk setiap individu
predictions_labels = [check_if_normal(pred) for pred in y_pred]

# Menampilkan hasil prediksi dengan status Normal atau Penyakit Mental
print("Prediksi dengan status Normal atau Penyakit Mental:")
print(predictions_labels[-5:])

# Menyimpan model untuk digunakan di aplikasi Streamlit atau aplikasi lainnya
model.save_model('mental_health_model.json')
joblib.dump(model, 'mental_health_model.pkl')  # Menyimpan model menggunakan joblib
