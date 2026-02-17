[<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/814/814256.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Smart Door Lock System with Liveness Detection</h3>

  <p align="center">
    Raspberry Pi 4 üzerinde çalışan, fotoğraf sahteciliğine karşı korumalı (Anti-Spoofing) ve internet gerektirmeyen uçtan uca biyometrik güvenlik sistemi.
    <br />
    <br />
    <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project/issues">Hata Bildir</a>
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
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 🚀 Proje Hakkında

Bu proje, geleneksel anahtar kullanımını ortadan kaldıran ve tamamen **Edge Computing (Uç Bilgi İşlem)** prensibiyle çalışan bir akıllı kapı kilidi sistemidir. Tüm işlemler Raspberry Pi üzerinde yerel olarak yapılır, hiçbir veri buluta gönderilmez.

Sistemi benzersiz kılan en önemli özellik, entegre **Liveness Detection (Canlılık Testi)** modülüdür. Bu sayede sistem, kapıyı açmaya çalışan kişinin gerçek bir insan mı yoksa bir fotoğraf/video mu olduğunu ayırt edebilir.

### 🔥 Temel Özellikler
* **🛡️ Anti-Spoofing Koruması:** Sahte yüz saldırılarını (telefondan fotoğraf gösterme, kağıt maske vb.) **%98 başarıyla** engeller.
* **🔒 Gizlilik ve Güvenlik:** KVKK uyumludur; biyometrik veriler cihaz dışına çıkmaz, şifreli olarak saklanır.
* **⚡ Özel Donanım Sürücüsü:** 3.3V GPIO sinyallerini 5V/12V kilit mekanizmasına güvenle iletmek için özel tasarlanmış **MOSFET** devresi içerir.
* **🚀 Hızlı Kayıt (One-Shot):** Yeni kullanıcı eklemek için modelin tekrar eğitilmesine gerek yoktur, işlem sadece 3 saniye sürer.

---

## 🧠 Sistem Mimarisi

Yazılım mimarisi, Raspberry Pi'nin sınırlı kaynaklarını verimli kullanmak üzere optimize edilmiştir. İşleyiş 3 ana aşamadan oluşur:

1.  **Yüz Tespiti (MTCNN):** Farklı ışık koşullarında yüzü ve 5 karakteristik noktayı (gözler, burun, ağız) tespit eder.
2.  **Canlılık Kontrolü (Anti-Spoofing):** Tespit edilen yüz TFLite modeline gönderilir. Eğer skor 0.5'in altındaysa "SAHTE" olarak işaretlenir ve erişim reddedilir.
3.  **Yüz Tanıma (FaceNet):** Canlı olduğu doğrulanan yüz, InceptionResNetV1 ile 512 boyutlu bir vektöre dönüştürülür ve veritabanıyla kıyaslanır.

---

## ⚡ Donanım Kurulumu

Proje, Raspberry Pi'yi yüksek akım çeken bir Solenoid Kilide bağlamak için özel bir devre tasarımı gerektirir.

| Bileşen | Özellik / Görev |
| :--- | :--- |
| **Raspberry Pi 4** | Ana İşlemci (4GB RAM önerilir) |
| **Pi Camera V3** | Yüksek çözünürlüklü görüntü alma (Sony IMX219) |
| **Solenoid Kilit** | 12V Elektromekanik Kapı Kilidi |
| **Röle Modülü** | 5V Tetiklemeli |
| **MOSFET (BS170)** | 3.3V Logic -> 5V Tetikleme Dönüştürücü |

> **Önemli:** Raspberry Pi GPIO pinleri doğrudan röleyi veya kilidi süremez. Tasarlanan MOSFET devresi, Pi'yi yüksek voltajdan korur.

---

## 📊 Performans Sonuçları

Sistem, laboratuvar ortamında ve gerçek dünya senaryolarında (farklı ışık, açı) test edilmiştir.

| Yöntem | Doğruluk (Accuracy) | Hız (FPS) | Açıklama |
| :--- | :--- | :--- | :--- |
| Haar Cascade | %88.0 | ~3.5 | Hızlı fakat hatalı |
| VGG16 | %92.0 | ~1.0 | Çok yavaş |
| **Bizim Sistem (FaceNet)** | **%96.5** | **~0.8** | **En iyi denge** |

* **Düşük Işık:** 100-200 lux ortamda %92.5 doğruluk.
* **Saldırı Engelleme:** 50 sahte giriş denemesinin 49'u başarıyla engellendi.

---

## 🛠️ Kurulum ve Kullanım

Projeyi kendi cihazınızda çalıştırmak için adımları takip edin:

1.  **Repoyu Klonlayın**
    ```bash
    git clone [https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git](https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git)
    cd Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project
    ```

2.  **Gerekli Kütüphaneleri Yükleyin**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı Başlatın**
    ```bash
    python main_gui.py
    ```

---

## 👨‍💻 Geliştirici

**Kemal Can Güngör**
* **Rol:** Sistem Mimarisi, Gömülü Yazılım & Donanım Tasarımı
* **Bağlantılar:** [LinkedIn](https://www.linkedin.com/in/kemal-can-güngör-4598b4234/) | [GitHub](https://github.com/KemalCaan)

---

<div align="center">
  <sub>© 2025 Kemal Can Güngör - İstanbul Kültür Üniversitesi Bitirme Projesi</sub>
</div>
](https://www.linkedin.com/in/kemal-can-g%C3%BCng%C3%B6r-4598b4234/)
