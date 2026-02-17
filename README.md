# Yüz Tanıma ve Canlılık Tespiti Tabanlı Akıllı Kapı Kilidi Sistemi
# (Smart Door Lock System Using Face Recognition & Liveness Detection)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red) ![Library](https://img.shields.io/badge/Library-PyTorch%20%7C%20OpenCV%20%7C%20TensorFlowLite-orange) ![License](https://img.shields.io/badge/license-MIT-green)

## 📄 Proje Özeti (Abstract)

Bu proje, **İstanbul Kültür Üniversitesi** Bilgisayar Mühendisliği bitirme projesi kapsamında geliştirilmiştir. [cite_start]Proje, **Raspberry Pi 4** üzerinde çalışan, derin öğrenme tabanlı ve temas gerektirmeyen bir biyometrik güvenlik sistemidir[cite: 26, 34].

[cite_start]Sistem, geleneksel yöntemlerin aksine sadece yüzü tanımakla kalmaz, aynı zamanda **TensorFlow Lite** tabanlı "Canlılık Tespiti" (Liveness Detection) modülü sayesinde fotoğraf veya video ile yapılan sahtecilik (spoofing) saldırılarını engeller[cite: 31, 39]. [cite_start]Veriler yerel olarak işlenir, herhangi bir bulut servisine ihtiyaç duymaz, bu da veri gizliliğini (KVKK/GDPR uyumlu) garanti altına alır[cite: 167].

---

## 🚀 Temel Özellikler

* [cite_start]**Yüksek Doğruluk:** MTCNN ve InceptionResNetV1 mimarileri kullanılarak **%96.5** doğruluk oranı elde edilmiştir[cite: 733].
* [cite_start]**Anti-Spoofing (Canlılık Testi):** Fotoğraf ve video ekranı gösterilerek yapılan sızma girişimlerini **%96** başarı oranıyla engeller[cite: 853].
* [cite_start]**Gömülü Sistem Mimarisi:** Harici bir GPU veya sunucuya ihtiyaç duymadan, tüm işlemler Raspberry Pi 4 CPU'su üzerinde gerçekleştirilir[cite: 848].
* [cite_start]**Otomatik Aydınlatma Dengeleme:** Düşük ışık koşullarında (100-200 lux) histogram eşitleme ile performans kaybını önler[cite: 432, 658].
* [cite_start]**Donanım Tabanlı Kilit Kontrolü:** Tanınan kullanıcı için GPIO pinleri üzerinden Solenoid kilidi tetikler[cite: 441].

---

## 🛠️ Kullanılan Teknolojiler ve Mimari

### Yazılım ve Yapay Zeka Modelleri
* [cite_start]**Yüz Tespiti ve Hizalama:** MTCNN (Multi-task Cascaded Convolutional Networks)[cite: 27].
* [cite_start]**Öznitelik Çıkarımı (Embedding):** InceptionResNetV1 (VGGFace2 üzerinde eğitilmiş) - 512 boyutlu vektör çıktısı verir[cite: 29, 438].
* [cite_start]**Canlılık Analizi (Liveness):** TensorFlow Lite (MobileNetV2 tabanlı hafif model)[cite: 439].
* [cite_start]**Karşılaştırma:** Cosine Similarity (Kosinüs Benzerliği) algoritması[cite: 440].
* [cite_start]**Arayüz (GUI):** Python Tkinter (Thread yapısı ile donma yapmayan arayüz)[cite: 433].

### Donanım Bileşenleri
* [cite_start]**Ana İşlemci:** Raspberry Pi 4 Model B (4GB RAM)[cite: 215].
* [cite_start]**Görüntüleme:** Raspberry Pi Camera Module V3 (Sony IMX219 Sensör)[cite: 232].
* [cite_start]**Kilit Mekanizması:** 12V Solenoid Kapı Kilidi[cite: 241].
* [cite_start]**Sürücü Devre:** 5V Röle Modülü ve 3.3V mantık seviyesini güvenli şekilde tetiklemek için tasarlanmış **BS170 MOSFET** devresi[cite: 405, 422].

---

## 🔄 Sistem Çalışma Akışı (Workflow)

[cite_start]Sistem gerçek zamanlı (Real-Time) olarak aşağıdaki döngüyü işletir[cite: 368, 477]:

1.  **Görüntü Alma:** Pi Kamera V3'ten ham görüntü alınır.
2.  **Ön İşleme:** Görüntü gri tonlamaya çevrilir ve Histogram Eşitleme uygulanır.
3.  **Yüz Tespiti:** MTCNN ile yüz bulunur ve 5 ana nokta (gözler, burun, dudak) işaretlenir.
4.  **Canlılık Kontrolü:** Kırpılan yüz TFLite modeline gönderilir.
    * *Sonuç "Fake" ise:* Erişim Reddedilir (Kilit açılmaz).
    * *Sonuç "Real" ise:* Bir sonraki adıma geçilir.
5.  **Vektör Çıkarımı:** InceptionResNetV1 yüzü 512 boyutlu sayısal bir vektöre dönüştürür.
6.  [cite_start]**Eşleştirme:** Veritabanındaki kayıtlı yüzlerle Kosinüs Benzerliği hesaplanır (Eşik Değeri > 0.90)[cite: 469].
7.  **Eylem:** Eşleşme varsa GPIO tetiklenir, röle çeker ve kapı açılır.

*(Buraya projenizdeki Workflow Diagram görselini ekleyebilirsiniz - Figure 8)*

---

## 📊 Performans ve Test Sonuçları

[cite_start]Proje, Haar Cascade ve VGG16 gibi geleneksel yöntemlerle kıyaslanmış ve 1.050 adet test senaryosu ile doğrulanmıştır[cite: 637, 728].

| Yöntem | Doğruluk (Accuracy) | Ortalama Hız (FPS) | Donanım |
| :--- | :---: | :---: | :---: |
| **MTCNN + InceptionResNetV1 (Bu Proje)** | **%96.5** | **~0.7 FPS** | **Raspberry Pi 4** |
| VGG16 + SSD300 | %92.0 | ~1.0 FPS | Raspberry Pi 4 |
| Haar Cascade + HOG | %88.0 | ~3.5 FPS | Raspberry Pi 4 |

* [cite_start]**Zorlu Koşullar:** Düşük ışıkta (<200 lux) ve 30 dereceye kadar açısal bozulmalarda dahi %90 üzeri başarı korunmuştur[cite: 658, 808].
* [cite_start]**Kaynak Tüketimi:** Çalışma esnasında CPU kullanımı ~%35, RAM kullanımı ~420MB seviyesindedir[cite: 872].

---

## 💻 Kurulum ve Kullanım

Projeyi yerel ortamınızda veya Raspberry Pi üzerinde çalıştırmak için:

1.  Repoyu klonlayın:
    ```bash
    git clone [https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git](https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git)
    cd Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project
    ```

2.  Gereksinimleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
    *(Gerekli kütüphaneler: torch, torchvision, facenet-pytorch, opencv-python, tensorflow-lite, RPi.GPIO)*

3.  Uygulamayı başlatın:
    ```bash
    python raspberry_gui.py
    ```

4.  **Kullanım:**
    * [cite_start]Arayüz üzerinden "Yeni Kişi Ekle" butonuna basarak 5-10 adet fotoğraf ile profil oluşturun[cite: 540].
    * [cite_start]Sistem otomatik olarak yüz profili vektörünü (`embeddings.pkl`) oluşturacaktır[cite: 566].
    * Ana ekrana dönüldüğünde sistem otomatik olarak tanıma moduna geçer.

---

## 👥 Proje Ekibi

* **Kemal Can Güngör** - [LinkedIn](https://www.linkedin.com/in/kemal-can-g%C3%BCng%C3%B6r-4598b4234/)
* **Ozan Emre Tunca** - [LinkedIn](https://www.linkedin.com/in/ozan-tunca-761b9a257/)

**Danışman:** Öğr. Gör. [cite_start]Basri Erdoğan [cite: 18]

---

## 📝 Lisans

Bu proje akademik amaçlarla geliştirilmiştir ve MIT Lisansı altındadır.
