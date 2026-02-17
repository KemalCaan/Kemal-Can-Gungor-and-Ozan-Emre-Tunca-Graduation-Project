# Yüz Tanıma ve Canlılık Tespiti Tabanlı Akıllı Kapı Kilidi Sistemi
# (Smart Door Lock System Using Face Recognition & Liveness Detection)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red) ![Library](https://img.shields.io/badge/Library-PyTorch%20%7C%20OpenCV%20%7C%20TensorFlowLite-orange) ![License](https://img.shields.io/badge/license-MIT-green)

## 📄 Proje Hakkında (About the Project)

Bu proje, **Elektrik-Elektronik Mühendisliği** bitirme tezi kapsamında **Kemal Can Güngör** tarafından geliştirilmiştir.

Proje, **Raspberry Pi 4** üzerinde çalışan, derin öğrenme tabanlı ve temas gerektirmeyen biometrik bir güvenlik sistemidir. Sistem, geleneksel yüz tanıma yöntemlerinin ötesine geçerek, **TensorFlow Lite** tabanlı "Canlılık Tespiti" (Liveness Detection) modülü sayesinde fotoğraf veya video ile yapılan sahtecilik (spoofing) saldırılarını engeller. Tüm veriler yerel olarak işlenir, herhangi bir bulut servisine ihtiyaç duyulmaz; bu da veri gizliliğini ve güvenliğini en üst düzeye çıkarır.

---

## 🚀 Temel Özellikler

* [cite_start]**Yüksek Doğruluk:** MTCNN ve InceptionResNetV1 mimarileri kullanılarak **%96.5** doğruluk oranı elde edilmiştir[cite: 733].
* [cite_start]**Anti-Spoofing (Canlılık Testi):** Fotoğraf ve video ekranı gösterilerek yapılan sızma girişimlerini **%96** başarı oranıyla engeller[cite: 853].
* [cite_start]**Gömülü Sistem Mimarisi:** Harici bir GPU veya sunucuya ihtiyaç duymadan, tüm işlemler Raspberry Pi 4 işlemcisi üzerinde gerçekleştirilir[cite: 848].
* [cite_start]**Otomatik Aydınlatma Dengeleme:** Düşük ışık koşullarında (100-200 lux) histogram eşitleme ile performans kaybını önler[cite: 457].
* [cite_start]**Donanım Tabanlı Kilit Kontrolü:** Tanınan kullanıcı için GPIO pinleri ve MOSFET sürücü devresi üzerinden Solenoid kilidi tetikler[cite: 405].

---

## 🛠️ Sistem Mimarisi ve Kullanılan Teknolojiler

Bu proje, donanım ve yazılım bileşenlerinin entegre çalıştığı bir gömülü sistem projesidir.

### 1. Donanım Bileşenleri (Hardware)
Proje aşağıdaki temel bileşenler üzerine kurulmuştur:
* [cite_start]**Ana İşlemci:** Raspberry Pi 4 Model B (4GB RAM)[cite: 215].
* [cite_start]**Görüntüleme:** Raspberry Pi Camera Module V3 (Sony IMX219 Sensör)[cite: 233].
* [cite_start]**Kilit Mekanizması:** 12V Solenoid Kapı Kilidi[cite: 241].
* [cite_start]**Sürücü Devre:** 3.3V mantık seviyesini 5V röleye güvenli şekilde iletmek için tasarlanmış **BS170 MOSFET** tabanlı anahtarlama devresi[cite: 405, 422].

*(Aşağıdaki alana projedeki donanım bağlantı şemasını -Figure 2- ekleyebilirsiniz)*
![Donanım Bağlantı Şeması](images/hardware_design.png)
*Şekil 1: Raspberry Pi, Röle ve Kilit Bağlantı Şeması*

### 2. Yazılım ve Yapay Zeka Modelleri (Software & AI)
* [cite_start]**Yüz Tespiti ve Hizalama:** MTCNN (Multi-task Cascaded Convolutional Networks)[cite: 314].
* [cite_start]**Öznitelik Çıkarımı (Embedding):** InceptionResNetV1 (VGGFace2 üzerinde eğitilmiş) - 512 boyutlu vektör çıktısı verir[cite: 275].
* [cite_start]**Canlılık Analizi (Liveness):** TensorFlow Lite (MobileNetV2 tabanlı hafif model)[cite: 463].
* [cite_start]**Karşılaştırma:** Cosine Similarity (Kosinüs Benzerliği) algoritması[cite: 348].

---

## 🔄 Çalışma Akışı (Workflow)

Sistem gerçek zamanlı (Real-Time) olarak aşağıdaki döngüyü işletir:

1.  **Görüntü Alma:** Pi Kamera V3'ten ham görüntü alınır.
2.  [cite_start]**Ön İşleme:** Görüntü gri tonlamaya çevrilir ve Histogram Eşitleme uygulanır[cite: 457].
3.  [cite_start]**Yüz Tespiti:** MTCNN ile yüz bulunur ve 5 ana nokta (gözler, burun, dudak) işaretlenir[cite: 460].
4.  **Canlılık Kontrolü:** Kırpılan yüz TFLite modeline gönderilir.
    * *Sonuç "Sahte" (Fake) ise:* Erişim Reddedilir (Kilit açılmaz).
    * [cite_start]*Sonuç "Gerçek" (Real) ise:* Bir sonraki adıma geçilir[cite: 464].
5.  [cite_start]**Vektör Çıkarımı:** InceptionResNetV1 yüzü 512 boyutlu sayısal bir vektöre dönüştürür[cite: 466].
6.  [cite_start]**Eşleştirme:** Veritabanındaki kayıtlı yüzlerle Kosinüs Benzerliği hesaplanır (Eşik Değeri > 0.90)[cite: 469].
7.  [cite_start]**Eylem:** Eşleşme varsa GPIO tetiklenir, röle çeker ve kapı açılır[cite: 471].

*(Aşağıdaki alana projedeki akış diyagramını -Figure 8- ekleyebilirsiniz)*
![Sistem Akış Diyagramı](images/workflow_diagram.png)
*Şekil 2: Yazılım Algoritma Akış Şeması*

---

## 📊 Performans Sonuçları

Proje, 1.050 adet test senaryosu ile doğrulanmış ve geleneksel yöntemlerle (Haar Cascade, VGG16) kıyaslanmıştır.

| Yöntem | Doğruluk (Accuracy) | Ortalama Hız (FPS) | Donanım |
| :--- | :---: | :---: | :---: |
| **MTCNN + InceptionResNetV1 (Bu Proje)** | **%96.5** | **~0.7 FPS** | **Raspberry Pi 4** |
| VGG16 + SSD300 | %92.0 | ~1.0 FPS | Raspberry Pi 4 |
| Haar Cascade + HOG | %88.0 | ~3.5 FPS | Raspberry Pi 4 |

* [cite_start]**Zorlu Koşullar:** Düşük ışıkta (<200 lux) ve 30 dereceye kadar açısal bozulmalarda dahi %90 üzeri başarı korunmuştur[cite: 808, 868].
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
    * Arayüz üzerinden "Yeni Kişi Ekle" butonuna basarak 5-10 adet fotoğraf ile profil oluşturun.
    * Sistem otomatik olarak yüz profili vektörünü (`embeddings.pkl`) oluşturacaktır.
    * Ana ekrana dönüldüğünde sistem otomatik olarak tanıma moduna geçer.

---

## 👤 Geliştirici

* **Kemal Can Güngör** - Elektrik-Elektronik Mühendisi
* [LinkedIn Profilim](https://www.linkedin.com/in/kemal-can-g%C3%BCng%C3%B6r-4598b4234/)
* [E-posta](mailto:kemalcangungor@hotmail.com)

---

## 📝 Lisans

Bu proje akademik amaçlarla geliştirilmiştir ve MIT Lisansı altındadır.
