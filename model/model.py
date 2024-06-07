import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Özellik çıkarma fonksiyonu
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Veri ve etiketlerin yüklenmesi
data_path = r'C:\Users\asus\Desktop\projem\data\TurEV-DB-master'
data = []
labels = []

# Veri setinin bulunduğu dizinlerde gezinip özellikleri çıkarma
for emotion in ['angry', 'calm', 'happy', 'sad']:
    emotion_path = os.path.join(data_path, emotion)
    if not os.path.exists(emotion_path):
        print(f"Dizin mevcut değil: {emotion_path}")
        continue
    for file_name in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file_name)
        if os.path.isfile(file_path):
            features = extract_features(file_path)
            data.append(features)
            labels.append(emotion)

X = np.array(data)
y = np.array(labels)

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Modeli ile eğitim (probability=True)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Modelin test edilmesi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Modeli kaydetme
model_path = r'C:\Users\asus\Desktop\projem\model\emotion_recognition_model.pkl'
joblib.dump(model, model_path)
print(f"Model kaydedildi: {model_path}")
