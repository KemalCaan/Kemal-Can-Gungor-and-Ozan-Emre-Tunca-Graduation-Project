<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/814/814256.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Smart Door Lock System with Liveness Detection</h3>

  <p align="center">
    Raspberry Pi 4 tabanlı, <b>Deep Learning</b> destekli, sahteciliğe karşı korumalı (Anti-Spoofing) ve internet gerektirmeyen uçtan uca biyometrik güvenlik sistemi.
    <br />
    <br />
    <a href="https://linkedin.com/in/kemal-can-güngör-4598b4234/">Geliştirici ile İletişime Geç</a>
    ·
    <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project/issues">Hata Bildir</a>
  </p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 🚀 Proje Hakkında

Bu proje, fiziksel anahtarların yerini alan, **Edge Computing (Uç Bilişim)** mimarisine sahip bir akıllı kapı kilidi sistemidir. Sistem, internet bağlantısına ihtiyaç duymadan tüm görüntü işleme ve yapay zeka modellerini Raspberry Pi 4 üzerinde yerel olarak çalıştırır.

En büyük inovasyonu, entegre **Liveness Detection (Canlılık Testi)** modülüdür. Birçok yüz tanıma sisteminin aksine, bu proje kapıyı açmaya çalışan kişinin gerçek bir insan mı yoksa bir fotoğraf/video mu olduğunu ayırt edebilir.

### 🔥 Temel Özellikler
* **🛡️ Anti-Spoofing Teknolojisi:** TFLite tabanlı model ile sahte yüz saldırılarını (fotoğraf, video, maske) **%98 başarıyla** engeller.
* **⚡ Yüksek Performans:** MTCNN ve FaceNet algoritmaları Raspberry Pi 4 için optimize edilmiştir; ortalama **0.7 FPS** hızında çalışır.
* **🔒 Gizlilik Odaklı:** Biyometrik veriler (yüz vektörleri) asla buluta gönderilmez, cihaz içinde şifreli (pickle) saklanır.
* **🔌 Özel Güç Sürücüsü:** 3.3V GPIO sinyallerini 12V kilit sistemine güvenle iletmek için **MOSFET** devresi tasarlanmıştır.

---

## 🧠 Yazılım Mimarisi ve Algoritma Akışı

Sistem, kameradan alınan her kareyi 4 aşamalı bir "Pipeline" üzerinden geçirir. Aşağıdaki şema sistemin karar mekanizmasını göstermektedir:

```mermaid
graph TD
    A["📷 Kamera Görüntüsü (Input)"] --> B{"Yüz Tespit Edildi mi?<br>(MTCNN)"}
    
    B -- Hayır --> A
    B -- Evet --> C{"🧟 Canlılık Testi<br>(TFLite Anti-Spoofing)"}
    
    C -- "SAHTE (Fotoğraf/Ekran)" --> D["❌ ERİŞİM REDDEDİLDİ<br>(Kırmızı Uyarı)"]
    
    C -- "GERÇEK İNSAN" --> E["🔢 Yüz Tanıma - Embedding<br>(InceptionResNetV1)"]
    
    E --> F{"Veritabanı Eşleşmesi<br>(Cosine Similarity > 0.9)"}
    
    F -- "Tanınmayan Kişi" --> D
    F -- "Kayıtlı Kullanıcı" --> G["✅ KAPI AÇILDI<br>(GPIO Röle Tetikleme)"]
    
    G --> H["Log Kaydı Tut"]
    D --> H
