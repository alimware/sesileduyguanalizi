
# Duygu Tanıma Projesi

## Genel Bakış
Bu proje, .wav ses dosyalarından (Öfkeli, Sakin, Mutlu, Üzgün) duyguları tanımayı amaçlamaktadır. Makine öğrenimi modeli kullanılarak duygular tanınmaktadır.

## Veri Seti
- **Ad:** TurEV-DB-master
- **Duygular:** Öfkeli (Angry), Sakin (Calm), Mutlu (Happy), Üzgün (Sad)
- **Dosya Formatı:** .wav
- **Örnek Sayısı:**
  - Öfkeli: 487
  - Sakin: 408
  - Mutlu: 357
  - Üzgün: 483

## Veri Ön İşleme ve Özellik Mühendisliği
- **Özellik Çıkarma:** Her bir ses dosyasından 40 MFCC özelliği çıkarılmış ve ortalama alınarak özellik vektörü oluşturulmuştur.
- **Veri Ayırma:** Veriler %80 eğitim ve %20 test olarak ayrılmıştır.

## Modelleme, Test ve Doğrulama
- **Model:** Doğrusal çekirdekli Destek Vektör Makinesi (SVM).
- **Eğitim:** Model, eğitim seti üzerinde eğitilmiştir.
- **Test:** Model, test seti üzerinde test edilmiş ve doğruluk skoru hesaplanmıştır.

## Dağıtım
- **Ortam:** Python 3.8+, GUI için Tkinter, ses işleme ve model eğitimi için pyaudio, librosa, sklearn, joblib.
- **API:** Eğitilmiş model joblib ile kaydedilmiş ve uygulamada tahminler için kullanılmıştır.

## Gerekli Kütüphaneler
Gerekli kütüphanelerin listesi `requirements.txt` dosyasında verilmiştir.

## Kurulum ve Çalıştırma Talimatları
1. Depoyu yerel makinenize klonlayın.
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt

Terminal açın. terminal split yapın. ilk terminale cd model yazın.
diğer terminale cd app yazın. ilk terminale python model.py yazın. çalışınca diğer terminal cd app.py yazın.
