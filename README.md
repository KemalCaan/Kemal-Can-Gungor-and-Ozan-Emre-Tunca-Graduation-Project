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

#### Donanım Bağlantı Şeması (Wiring Diagram)
Aşağıdaki şema, sistemin elektriksel bağlantı yapısını göstermektedir:


graph LR
    RPI[Raspberry Pi 4 GPIO] -- 3.3V Sinyal --> MOSFET[BS170 MOSFET Sürücü]
    MOSFET -- 5V Tetikleme --> RELAY[5V Röle Modülü]
    PSU[12V Güç Kaynağı] -- Güç Beslemesi --> RELAY
    RELAY -- Anahtarlama --> LOCK[Solenoid Kilit]
    CAM[Pi Camera V3] -- CSI Arayüzü --> RPI


2. Yazılım Teknolojileri (Software Stack)
Dil: Python 3.x

Çerçeveler: PyTorch (FaceNet), TensorFlow Lite (Anti-Spoofing)

Görüntü İşleme: OpenCV (cv2)

Arayüz: Tkinter (Multithreaded GUI)




graph TD
    %% Başlangıç
    Start([Kamera Görüntüsü Alımı]) --> PreProc[Ön İşleme: Gri Tonlama & Histogram Eşitleme]
    %% Aşama 1: Tespit
    PreProc --> Detect{YÜZ TESPİTİ\n(MTCNN)}
    Detect -- Yüz Yok --> Start
    Detect -- Yüz Var --> Crop[Yüzü Kırp ve Hizala]
    %% Aşama 2: Güvenlik Kontrolü (Canlılık)
    Crop --> Liveness{CANLILIK TESTİ\n(TFLite Model)}
    Liveness -- SAHTE (Spoof) --> Deny[ERİŞİM REDDİ\n(Kırmızı LED Uyarısı)]    
    %% Aşama 3: Tanıma
    Liveness -- GERÇEK (Real) --> Embed[Vektör Çıkarımı\n(InceptionResNetV1)]
    Embed --> Match[Veritabanı Karşılaştırma\n(Cosine Similarity)]
    %% Aşama 4: Karar ve Eylem
    Match --> Decision{Benzerlik > 0.90?}
    Decision -- Hayır (Tanınmadı) --> Deny
    Decision -- Evet (Tanındı) --> Actuate[GPIO & MOSFET Tetikleme]
    Actuate --> Unlock(((KİLİT AÇILDI)))    
    %% Döngü
    Deny --> Start
    Unlock -- 5 Sn Sonra Kitle --> Start
    %% Stil Tanımlamaları
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:1px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;
    classDef resultDone fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef resultFail fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    class PreProc,Crop,Embed,Match,Actuate process;
    class Detect,Liveness,Decision decision;
    class Unlock resultDone;
    class Deny resultFail;































