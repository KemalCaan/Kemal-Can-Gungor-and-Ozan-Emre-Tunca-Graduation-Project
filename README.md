# Yüz Tanıma ve Canlılık Tespiti Tabanlı Gömülü Akıllı Kilit Sistemi
# (Smart Door Lock System Using Face Recognition & Liveness Detection)

![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red?style=for-the-badge&logo=raspberrypi)
![Language](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/PyTorch-FaceNet-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Proje Hakkında (Project Abstract)

Bu çalışma, **Elektrik-Elektronik Mühendisliği** bitirme tezi kapsamında **Kemal Can Güngör** tarafından tasarlanmış ve gerçekleştirilmiştir.

Proje, biyometrik güvenlik sistemlerinin gömülü sistemler üzerinde gerçek zamanlı uygulanabilirliğini göstermeyi amaçlayan, **Derin Öğrenme (Deep Learning)** tabanlı temassız bir erişim kontrol sistemidir. Sistem, **Raspberry Pi 4** mimarisi üzerinde çalışmakta olup, **TensorFlow Lite** tabanlı "Canlılık Tespiti" (Anti-Spoofing) algoritması ile fotoğraf veya video ekranı kullanılarak yapılan sızma girişimlerini engellemektedir.

Geleneksel sistemlerden farklı olarak, 3.3V mantık seviyesine sahip mikroişlemci ile 12V endüstriyel kilit mekanizması arasında güvenli anahtarlama sağlayan özel bir **MOSFET sürücü devresi** tasarlanmış ve entegre edilmiştir.

---

## ⚙️ Teknik Özellikler ve Performans

Sistem, laboratuvar ortamında farklı aydınlatma ve açı koşullarında 1.050 adet test senaryosu ile doğrulanmıştır.

| Parametre | Değer / Açıklama |
| :--- | :--- |
| **İşlemci Mimarisi** | ARM Cortex-A72 (Raspberry Pi 4) |
| **Yüz Tanıma Modeli** | InceptionResNetV1 (Pre-trained VGGFace2) |
| **Yüz Tespit Algoritması** | MTCNN (Multi-task Cascaded Convolutional Networks) |
| **Doğruluk Oranı (Accuracy)** | **%96.5** |
| **Canlılık Tespiti (Spoofing)** | **%96 Başarım** (Fotoğraf/Video Engelleme) |
| **Tepki Süresi** | ~0.7 FPS (CPU Tabanlı Gerçek Zamanlı İşleme) |
| **Güç Tüketimi** | Aktif: ~6.5W / Bekleme: ~2.7W |

---

## 🛠️ Sistem Mimarisi (Hardware & Software)

Proje, görüntü işleme yazılımı ile elektromekanik kontrol donanımının entegrasyonundan oluşmaktadır.

### 1. Donanım Katmanı (Hardware Layer)
* **Ana Kontrol Ünitesi:** Raspberry Pi 4 Model B (4GB RAM).
* **Görüntüleme:** Raspberry Pi Camera Module V3 (Sony IMX219 Sensör).
* **Eyleyici (Actuator):** 12V Solenoid Kapı Kilidi.
* **Güç Elektroniği:** Raspberry Pi'nin 3.3V GPIO çıkışını korumak ve 5V Röle modülünü kararlı tetiklemek için tasarlanmış **BS170 MOSFET** tabanlı lojik seviye dönüştürücü ve sürücü devresi.

#### Donanım Bağlantı Şeması
```mermaid
graph LR
    RPI[Raspberry Pi 4 GPIO] -- 3.3V Sinyal --> MOSFET[BS170 MOSFET Sürücü]
    MOSFET -- 5V Tetikleme --> RELAY[5V Röle Modülü]
    PSU[12V Güç Kaynağı] -- Güç Beslemesi --> RELAY
    RELAY -- Anahtarlama --> LOCK[Solenoid Kilit]
    CAM[Pi Camera V3] -- CSI Arayüzü --> RPI

flowchart TD
    A([Başlangıç / Kamera Görüntüsü]) --> B[Ön İşleme: Gri Tonlama & Histogram Eşitleme]
    B --> C{Yüz Tespiti / MTCNN}
    C -- Yüz Bulunamadı --> A
    C -- Yüz Tespit Edildi --> D[Canlılık Testi / Anti-Spoofing]
    
    D -- SAHTE (Fotoğraf/Video) --> E[Erişim Reddedildi / Kırmızı LED]
    D -- GERÇEK (Canlı Kişi) --> F[Öznitelik Çıkarımı / InceptionResNetV1]
    
    F --> G[Vektör Karşılaştırma / Cosine Similarity]
    G --> H{Benzerlik > 0.90?}
    
    H -- Hayır --> E
    H -- Evet --> I[GPIO Tetikleme / MOSFET Sürüşü]
    I --> J([Kilit Açıldı / Yeşil LED])
    E --> A
    J --> A



Yöntem,Doğruluk (Accuracy),Donanım Yükü,Sonuç
Bu Çalışma,%96.5,Orta,Yüksek Güvenlik & Kararlılık
VGG16 + SSD,%92.0,Çok Yüksek,Yavaş Çalışma (<1 FPS)
Haar Cascade,%88.0,Düşük,Düşük Doğruluk & Hatalı Tespit


git clone [https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git](https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git)
cd Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python raspberry_gui.py

👨‍💻 Geliştirici Bilgileri
Kemal Can Güngör

Unvan: Elektrik-Elektronik Mühendisi

Uzmanlık: Gömülü Sistemler, IoT, Bilgisayarlı Görü (Computer Vision)

İletişim: LinkedIn Profilim | E-posta Gönder




