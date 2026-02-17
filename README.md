<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/814/814256.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Smart Door Lock System with Liveness Detection</h3>

  <p align="center">
    Raspberry Pi 4 tabanlı, <b>Anti-Spoofing (Sahtecilik Koruması)</b> özellikli ve internet gerektirmeyen uçtan uca biyometrik güvenlik sistemi.
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

Bu proje, geleneksel anahtar sistemlerini ortadan kaldıran, **Edge Computing (Uç Bilişim)** mimarisiyle tasarlanmış akıllı bir güvenlik sistemidir. Sistem, internet bağlantısına ihtiyaç duymadan tüm yapay zeka işlemlerini Raspberry Pi üzerinde yerel olarak gerçekleştirir.

Piyasadaki birçok yüz tanıma sisteminin aksine, bu proje sadece yüzü tanımakla kalmaz; **"Bu bir gerçek insan mı yoksa telefon ekranı/fotoğraf mı?"** sorusunu sorarak sahtecilik girişimlerini engeller.

### 🔥 Temel Özellikler
* **🛡️ Anti-Spoofing (Canlılık Testi):** TFLite modeli sayesinde fotoğrafla veya videoyla yapılan kilit açma girişimlerini **%98 başarıyla** engeller.
* **⚡ Yüksek Performans:** MTCNN ve FaceNet algoritmaları Raspberry Pi 4 için optimize edilmiştir; 1.4 saniye içinde karar verir.
* **🔒 KVKK Uyumlu:** Biyometrik veriler asla buluta gönderilmez, cihaz içerisinde şifreli (pickle) olarak saklanır.
* **🔌 Özel Güç Sürücüsü:** Raspberry Pi'yi yüksek voltajdan korumak için özel tasarlanmış MOSFET devresi kullanır.

---

## ⚙️ Sistem Nasıl Çalışır? (Çalışma Mantığı)

Sistemin karar verme mekanizması adım adım aşağıdaki gibidir. Her kare (frame) bu güvenlik süzgecinden geçirilir:

```mermaid
graph TD
    A["📷 Kamera Görüntüsü Alınır"] --> B{"Yüz Tespit Edildi mi?"}
    
    B -- Hayır --> A
    B -- "Evet (MTCNN)" --> C{"🧟 Canlılık Kontrolü<br>(Spoof or Real?)"}
    
    C -- "SAHTE (Fotoğraf/Ekran)" --> D["❌ ERİŞİM REDDEDİLDİ<br>(Kırmızı Uyarı)"]
    
    C -- "GERÇEK İNSAN" --> E["👤 Kimlik Doğrulama<br>(FaceNet Embedding)"]
    
    E --> F{"Veritabanı Eşleşmesi<br>(Benzerlik > %90)"}
    
    F -- "Tanınmayan Kişi" --> D
    F -- "Kayıtlı Kullanıcı" --> G["✅ KAPI AÇILDI<br>(Röle Tetiklenir)"]
    
    G --> H["Log Kaydı Tutulur"]
    D --> H
