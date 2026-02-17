<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/814/814256.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Smart Door Lock with Liveness Detection</h3>

  <p align="center">
    Raspberry Pi 4 tabanlı, fotoğraf sahteciliğine karşı korumalı (Anti-Spoofing) ve internet gerektirmeyen biyometrik güvenlik sistemi.
    <br />
    <br />
    <a href="#demo">Demo Görüntüle</a>
    ·
    <a href="https://linkedin.com/in/kemal-can-güngör-4598b4234/">İletişime Geç</a>
  </p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 🚀 Proje Hakkında (About The Project)

Bu proje, geleneksel anahtar sistemlerini ortadan kaldıran, tamamen **çevrimdışı (offline)** çalışan ve "Edge Computing" prensibiyle tasarlanmış bir akıllı kapı kilidi sistemidir.

Benzer sistemlerden en büyük farkı; **Liveness Detection (Canlılık Testi)** modülüne sahip olmasıdır. Bu sayede kapı kilidi, kişinin fotoğrafı veya videosu gösterilerek kandırılamaz.

### 🔥 Temel Özellikler
* ✅ **Anti-Spoofing:** Sahte yüzleri (fotoğraf/ekran) **%98 başarıyla** engeller.
* ✅ **Gizlilik Odaklı:** Biyometrik veriler buluta gitmez, sadece Raspberry Pi üzerinde şifreli saklanır.
* ✅ **Özel Donanım Sürücüsü:** 3.3V GPIO sinyallerini 5V röleye güvenle iletmek için kendi tasarladığım MOSFET devresi kullanılır.
* ✅ **Hızlı Kayıt:** Yeni bir kullanıcıyı sisteme eklemek sadece 3 saniye sürer (Model eğitimine gerek duymaz).

---

## 🧠 Sistem Mimarisi (System Workflow)

Sistemin karar verme mekanizmasını gösteren akış şeması aşağıdadır. (GitHub üzerinde otomatik render edilir).

```mermaid
graph TD
    A[Kamera Görüntüsü] --> B{Yüz Tespit Edildi mi?}
    B -- Hayır --> A
    B -- Evet (MTCNN) --> C{Canlılık Testi}
    
    C -- SAHTE (Fotoğraf/Video) --> D[❌ Erişim Reddedildi]
    D --> E[Log Kaydı Tut]
    
    C -- GERÇEK KİŞİ --> F[Yüz Tanıma - Embedding]
    F --> G{Veritabanı Eşleşmesi}
    
    G -- Eşleşme Yok (< %90) --> D
    G -- Eşleşme Var (> %90) --> H[✅ KAPI AÇILDI]
    H --> I[Röle Tetiklenir (GPIO)]
    I --> E
