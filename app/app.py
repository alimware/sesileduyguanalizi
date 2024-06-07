import tkinter as tk
from tkinter import messagebox, filedialog
import os
import numpy as np
import librosa
import joblib
import pyaudio
import wave
import threading

class EmotionPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Predictor")
        self.root.geometry("400x450")
        
        self.label = tk.Label(root, text="Duygu Analizi", font=("Helvetica", 16))
        self.label.pack(pady=10)
        
        self.emotion_label = tk.Label(root, text="Seslendireceğiniz Duyguyu Seçin:", font=("Helvetica", 12))
        self.emotion_label.pack(pady=5)
        
        self.emotions = ["angry", "happy", "sad", "calm"]
        self.selected_emotion = tk.StringVar()
        self.selected_emotion.set(self.emotions[0])
        
        self.emotion_menu = tk.OptionMenu(root, self.selected_emotion, *self.emotions)
        self.emotion_menu.pack(pady=5)
        
        self.record_button = tk.Button(root, text="Ses Kaydı Başlat", command=self.start_recording)
        self.record_button.pack(pady=10)
        
        self.analyze_button = tk.Button(root, text="Kaydedilen Ses Dosyasını Analiz Et", command=self.analyze_recorded_audio, state=tk.DISABLED)
        self.analyze_button.pack(pady=10)
        
        self.play_button = tk.Button(root, text="Kaydedilen Ses Dosyasını Oynat", command=self.play_recorded_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=20)
        
        self.score_label = tk.Label(root, text="Skor: 0", font=("Helvetica", 12))
        self.score_label.pack(pady=20)
        
        self.audio_file_path = None
        self.model_path = r"C:\Users\asus\Desktop\projem\model\emotion_recognition_model.pkl"  # Model dosyasının yolunu belirtin
        self.audio_frames = []
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.audio_duration = 3
        self.score = 0
    
    def start_recording(self):
        self.record_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.result_label.config(text="Ses kaydediliyor...")
        
        # Ses kaydı işlemi
        self.audio_frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format, channels=self.channels, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        
        for _ in range(0, int(self.sample_rate / self.chunk_size * self.audio_duration)):
            data = stream.read(self.chunk_size)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        self.result_label.config(text="Ses kaydedildi. Kaydedilen ses dosyasını analiz etmek veya oynatmak için ilgili düğmelere basın.")
        self.analyze_button.config(state=tk.NORMAL)
        self.play_button.config(state=tk.NORMAL)
        
        # Ses dosyasını .wav olarak kaydetme
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")])
        if file_path:
            self.save_audio_to_wav(file_path)
    
    def save_audio_to_wav(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        self.audio_file_path = file_path
    
    def analyze_recorded_audio(self):
        try:
            if not self.audio_file_path:
                self.audio_file_path = filedialog.askopenfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")])
                if not self.audio_file_path:
                    messagebox.showwarning("Uyarı", "Lütfen bir ses dosyası seçin.")
                    return
            
            # Ses dosyasını yükle ve özelliklerini çıkar
            features = self.extract_features(self.audio_file_path)
            
            # Eğitilmiş modeli yükle
            model = joblib.load(self.model_path)
            
            # Duygu tahmini yap
            features = np.array(features).reshape(1, -1)  # Ensure the features are a 2D array
            emotion_probabilities = model.predict_proba(features)[0]
            predicted_emotion = model.classes_[np.argmax(emotion_probabilities)]
            predicted_emotion_probability = np.max(emotion_probabilities)
            
            # Seçilen doğru duygu
            true_emotion = self.selected_emotion.get()
            
            # Sonucu göster
            self.result_label.config(text=f"Analiz Sonucu: {predicted_emotion} ({predicted_emotion_probability*100:.2f}%)")
            
            # Skor güncelle
            if predicted_emotion == true_emotion:
                score_increment = int(predicted_emotion_probability * 10)
                self.score += score_increment
                self.score_label.config(text=f"Skor: {self.score}")
            else:
                messagebox.showinfo("Sonuç", f"Tahmin yanlıştı. Doğru duygu: {true_emotion}")
        except Exception as e:
            messagebox.showerror("Hata", f"Duygu analizi sırasında bir hata oluştu: {str(e)}")
    
    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    
    def play_recorded_audio(self):
        try:
            if not self.audio_file_path:
                self.audio_file_path = filedialog.askopenfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")])
                if not self.audio_file_path:
                    messagebox.showwarning("Uyarı", "Lütfen bir ses dosyası seçin.")
                    return
            
            threading.Thread(target=self.play_audio_thread).start()
        except Exception as e:
            messagebox.showerror("Hata", f"Ses dosyasını oynatırken bir hata oluştu: {str(e)}")
    
    def play_audio_thread(self):
        try:
            # WAV dosyasını oynatma
            wf = wave.open(self.audio_file_path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            
            data = wf.readframes(self.chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(self.chunk_size)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            messagebox.showerror("Hata", f"Ses dosyasını oynatırken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionPredictorApp(root)
    root.mainloop()
