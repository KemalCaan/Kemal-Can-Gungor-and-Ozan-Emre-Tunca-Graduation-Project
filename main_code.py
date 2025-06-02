import os
import shutil
import pickle
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import RPi.GPIO as GPIO
import time
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageTk

# Ensure dataset directory exists
ROOT_DATASET = "dataset"
if not os.path.exists(ROOT_DATASET):
    os.makedirs(ROOT_DATASET)

# Paths
EMBEDDINGS_PATH = "embeddings.pkl"
LOGS_PATH = "actions.log"

# GPIO setup
RELAY_PIN = 20
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)



# Models
interpreter = tflite.Interpreter(model_path="FaceAntiSpoofing.tflite")
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# --- Dosyanın en başında, modelleri yükledikten hemen sonra ekleyin ---
# interpreter, inp_det, out_det zaten global olarak tanımlı
def is_live(face_bgr):
    """
    Anti-spoof testi: face_bgr BGR formatında yüz kırpıntısı
    dönerse True ise gerçek, False ise spoof.
    """
    # input boyutlarını al
    _, in_h, in_w, _ = inp_det['shape']
    # ön işleme
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (in_w, in_h))
    inp = np.expand_dims(face.astype(np.float32) / 255.0, 0)
    # tflite yorumlayıcıya ver
    interpreter.set_tensor(inp_det['index'], inp)
    interpreter.invoke()
    score = interpreter.get_tensor(out_det['index'])[0][0]
    # score < 0.5 gerçek
    return score < 0.5



try:
    detector = MTCNN(image_size=160, margin=0, keep_all=True)
    aligner  = MTCNN(image_size=160, margin=0, keep_all=False)
    resnet   = InceptionResnetV1(pretrained='vggface2').eval()
except Exception as e:
    messagebox.showerror("Model Hatası", f"Model yüklenemedi:\n{e}")
    raise SystemExit

def create_folder(name):
    folder = os.path.join(ROOT_DATASET, name)
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Dosya Hatası", f"Klasör oluşturulamadı:\n{e}")
        return None
    return folder

def get_face_tensor(rgb_image):
    try:
        aligned = aligner(rgb_image)
    except Exception:
        return None
    if aligned is None:
        return None
    t = aligned
    if t.ndim == 5:
        t = t[0]
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        return None
    return t

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Akıllı Kapı Kilidi Sistemi")
        self.geometry("1280x720")
        self.resizable(False, False)

        # Load embeddings & logs
        self.embeddings = []
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH,'rb') as f:
                self.embeddings = pickle.load(f)
        self.logs = []
        if os.path.exists(LOGS_PATH):
            with open(LOGS_PATH,'r') as f:
                self.logs = f.read().splitlines()

        # Camera setup
        try:
            self.cap = Picamera2()
            cfg = self.cap.create_video_configuration(main={"format":"BGR888","size":(640,480)})
            self.cap.configure(cfg)
            try:
                self.cap.set_controls({controls.AwbEnable: True})
            except Exception:
                pass
            self.cap.start()
        except Exception as e:
            messagebox.showerror("Kamera Hatası", f"{e}")
            self.destroy()
            return

        # Frames
        self.frames = {}
        for F in (MainMenu, NewPersonMenu, DeletePersonMenu, VerifyMenu, LogMenu):
            frm = F(self)
            self.frames[F] = frm
            frm.place(relwidth=1, relheight=1)
        self.show_frame(MainMenu)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_frame(self, cls):
        frame = self.frames[cls]
        frame.tkraise()
        # Eğer LogMenu gösteriliyorsa verileri güncelle
        if isinstance(frame, LogMenu):
            frame.refresh(self.logs)


    def save_embeddings(self):
        with open(EMBEDDINGS_PATH,'wb') as f:
            pickle.dump(self.embeddings, f)

    def log_action(self, text):
        entry = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}"
        self.logs.append(entry)
        with open(LOGS_PATH,'a') as f:
            f.write(entry + "\n")

    def on_closing(self):
        # 1) Önce log dosyasını yedek klasöre taşı
        backup_dir = "logs_backup"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"actions_{timestamp}.log")
        try:
            shutil.move(LOGS_PATH, backup_path)
        except FileNotFoundError:
            pass

        # 2) Arayüzdeki logları temizle
        self.logs.clear()
        # Eğer LogMenu açıksa hemen temizle
        self.frames[LogMenu].refresh(self.logs)

        # 3) Kamera ve GPIO kapat
        try:
            self.cap.stop()
        except:
            pass
        GPIO.cleanup()
        self.destroy()


class MainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # Başlık
        tk.Label(self, text="FACE RECOGNITION SYSTEM", font=("Arial", 32, "bold"))\
            .place(relx=0.5, y=50, anchor="center")

        # 3 Ana buton: Kontrol Et, Kişi Ekle, Kişi Sil
        btn_w, btn_h = 300, 60
        y_pos = 200
        ttk.Button(self, text="Kontrol Et",
                   command=lambda: master.show_frame(VerifyMenu))\
            .place(x=100, y=y_pos, width=btn_w, height=btn_h)
        ttk.Button(self, text="Kişi Ekle",
                   command=lambda: master.show_frame(NewPersonMenu))\
            .place(x=(1280-btn_w)//2, y=y_pos, width=btn_w, height=btn_h)
        ttk.Button(self, text="Kişileri Görüntüle",
                   command=lambda: master.show_frame(DeletePersonMenu))\
            .place(x=1280-100-btn_w, y=y_pos, width=btn_w, height=btn_h)

        # Çıkış butonu
        ttk.Button(self, text="Çıkış", command=master.on_closing)\
            .place(relx=0.5, y= y_pos+150, width=200, height=50, anchor="n")

        # Loglar butonu (sağ alt köşe)
        ttk.Button(self, text="Loglar",
                   command=lambda: master.show_frame(LogMenu))\
            .place(x=1280-150, y=720-70, width=120, height=40)

class NewPersonMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # Başlık
        tk.Label(self, text="Yeni Kişi Ekle", font=("Arial", 28))\
            .place(relx=0.5, y=40, anchor="center")

        # Sağda iki buton: Fotoğraf Çekerek, Dosyadan Seç
        btn_w, btn_h = 250, 50
        x_right = 1280 - 100 - btn_w
        ttk.Button(self, text="📷 Fotoğraf Çekerek", command=self.capture_photo)\
            .place(x=x_right, y=150, width=btn_w, height=btn_h)
        ttk.Button(self, text="📂 Dosyadan Seç", command=self.select_photo)\
            .place(x=x_right, y=230, width=btn_w, height=btn_h)

        # Geri Dön butonu (alt orta)
        ttk.Button(self, text="← Geri Dön",
                   command=lambda: master.show_frame(MainMenu))\
            .place(relx=0.5, y=720-80, width=200, height=40, anchor="center")

        
        
    def _prompt_name(self):
        name = simpledialog.askstring("Kayıt", "Kişi Adı:", parent=self)
        if not name:
            return None
        # aynı isim embeddings veya klasörde var mı?
        if any(name == e[0] for e in self.master.embeddings) or \
           os.path.isdir(os.path.join(ROOT_DATASET, name)):
            messagebox.showerror("Uyarı", f"\"{name}\" zaten kayıtlı.")
            return None
        return name


    def capture_photo(self):
        win = "Yeni Kişi Ekle"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)

        prev = time.time()
        cnt = 0
        fps = 0
        boxes = None

        while True:
            raw = self.master.cap.capture_array()
            rgb = raw
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            disp = bgr.copy()

            cnt += 1
            if cnt % 3 == 0:
                now = time.time()
                fps = 1.0 / (now - prev)
                prev = now
            cv2.putText(disp, f"FPS:{fps:.1f}", (480,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if cnt % 3 == 0:
                boxes, _ = detector.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                captured = (raw.copy(), rgb.copy())
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return
        cv2.destroyAllWindows()
        bgr_cap, rgb_cap = captured
        boxes, _ = detector.detect(rgb_cap)
        if boxes is None or len(boxes) == 0:
            messagebox.showerror("Hata", "Yüz bulunamadı.")
            return
        x1, y1, x2, y2 = map(int, boxes[0])
        face_bgr = bgr_cap[y1:y2, x1:x2]
        if face_bgr.size == 0 or is_live(face_bgr):
            messagebox.showerror("Spoof", "Bu gerçek bir insan değil.")
            return
        
        name = simpledialog.askstring("Kayıt","Kişi Adı:",parent=self)
        if not name or name in [e[0] for e in self.master.embeddings]:
            messagebox.showwarning("Uyarı","Geçerli bir isim girin."); return
        folder = create_folder(name)
        fn = f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(os.path.join(folder,fn), bgr_cap)

        face = get_face_tensor(rgb_cap)
        if face is None:
            messagebox.showerror("Hata","Yüz align edilemedi"); return
        emb = resnet(face.unsqueeze(0))[0].detach()
        self.master.embeddings.insert(0,(name,emb))
        self.master.save_embeddings()
        self.master.log_action(f"Kayıt:{name}")
        self.master.frames[DeletePersonMenu].populate()
        messagebox.showinfo("Başarılı",f"{name} kaydedildi")

    def select_photo(self):
        path = filedialog.askopenfilename(
            title="Resim Seç",
            initialdir=".",
            filetypes=[("Resim", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return

        # Dataset klasöründeki fotoğrafları tekrar kullanmayı engelle
        abs_dataset = os.path.abspath(ROOT_DATASET) + os.sep
        if os.path.abspath(path).startswith(abs_dataset):
            messagebox.showerror("Uyarı", "Bu fotoğraf daha önce kullanılmış, lütfen başka bir tane deneyin.")
            return

        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Hata", "Dosya açılamadı.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Yüz tespiti
        boxes, _ = detector.detect(rgb)
        if boxes is None or len(boxes) == 0:
            messagebox.showerror("Hata", "Yüz bulunamadı.")
            return

        x1, y1, x2, y2 = map(int, boxes[0])
        face_bgr = bgr[y1:y2, x1:x2]
        if face_bgr.size == 0 or is_live(face_bgr):
            messagebox.showerror("Spoof","Bu gerçek bir insan değil."); return

        # Align kontrolü
        face = get_face_tensor(rgb)
        if face is None:
            messagebox.showerror("Hata", "Yüz align edilemedi.")
            return

        # İsim sor ve daha önce var mı kontrol et
        name = self._prompt_name()
        if not name:
            return

        # Kaydet
        folder = create_folder(name)
        fn = f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(os.path.join(folder, fn), bgr)

        emb = resnet(face.unsqueeze(0))[0].detach()
        self.master.embeddings.insert(0, (name, emb))
        self.master.save_embeddings()
        self.master.log_action(f"Kayıt (dosya): {name}")
        self.master.frames[DeletePersonMenu].populate()
        messagebox.showinfo("Başarılı", f"{name} kaydedildi")


class DeletePersonMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        # Başlık
        tk.Label(self, text="Kişiyi Görüntüle", font=("Arial", 28, "bold"))\
          .pack(pady=(20, 10))

        # Kişi satırlarını koyacağımız container
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True, padx=20, pady=(0,10))

        # Geri Dön butonu
        ttk.Button(self, text="← Geri Dön",
                   command=lambda: master.show_frame(MainMenu))\
           .pack(pady=(0,20))

        # Başlangıçta listele
        self.populate()


    def populate(self):
        # Önce eski satırları temizleyelim
        for w in self.container.winfo_children():
            w.destroy()

        # Her embedding için bir satır
        for idx, (name, _) in enumerate(self.master.embeddings):
            # Satırın kendisi: beyaz, ince solid border
            row = tk.Frame(self.container,
                           bg="white",
                           bd=1,
                           relief="solid")
            row.pack(fill="x", pady=5)

            # İsmi ortalayacak Label
            lbl = tk.Label(row,
                           text=name,
                           bg="white",
                           font=("Arial", 16))
            lbl.pack(side="left", expand=True, fill="x", padx=10, pady=5)

            # Sil butonu
            btn = tk.Button(row,
                            text="Sil",
                            fg="red",
                            bd=0,
                            command=lambda i=idx: self.delete(i))
            btn.pack(side="right", padx=10, pady=5)

    def delete(self, idx):
        name, _ = self.master.embeddings[idx]
        if messagebox.askyesno("Sil", f"{name} silinsin mi?"):
            shutil.rmtree(os.path.join(ROOT_DATASET, name), ignore_errors=True)
            self.master.embeddings.pop(idx)
            self.master.save_embeddings()
            self.master.log_action(f"Silme:{name}")
            self.populate()

class VerifyMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        # Başlık ortada, büyük puntoda
        tk.Label(self, text="Kontrol Et", font=("Arial", 28, "bold"))\
          .place(relx=0.5, y=40, anchor="center")

        # Buton boyutları ve sağdaki X koordinatı
        btn_w, btn_h = 250, 50
        x_right = 1280 - 100 - btn_w  # sağ kenardan 100px içeride

        # “Başlat” butonu
        ttk.Button(self, text="▶ Başlat",
                   command=self.verify_once)\
          .place(x=x_right, y=150, width=btn_w, height=btn_h)

        # “Geri Dön” butonu
        ttk.Button(self, text="← Geri Dön",
                   command=lambda: master.show_frame(MainMenu))\
          .place(x=x_right, y=230, width=btn_w, height=btn_h)


    def verify_once(self):
        win = "Kişi Tanıma"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)

        last_boxes = []
        last_probs = []
        frame_idx = 0
        prev_time = time.time()
        fps = 0.0


        GPIO.setmode(GPIO.BCM)
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        door_opened = False

        # Anti-spoof TFLite interpreter ön yükleme
        interpreter = tflite.Interpreter(model_path="FaceAntiSpoofing.tflite")
        interpreter.allocate_tensors()
        inp_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]

        def is_live(face_bgr):
            _, in_h, in_w, _ = inp_det['shape']
            face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (in_w, in_h))
            inp = np.expand_dims(face.astype(np.float32) / 255.0, 0)
            interpreter.set_tensor(inp_det['index'], inp)
            interpreter.invoke()
            score = interpreter.get_tensor(out_det['index'])[0][0]
            return score < 0.5

        prev_recog = None  # None=henüz log yazılmadı, True=son durum tanındı, False=son durum tanınmadı

        while True:
            # pencere kapandı mı diye kontrol
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break

            raw = self.master.cap.capture_array()        # RGB frame
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)   # BGR gösterim
            disp = bgr.copy()

            # FPS hesaplama
            now = time.time()
            fps = 1.0/(now - prev_time) if now != prev_time else fps
            prev_time = now
            cv2.putText(disp, f"FPS: {fps:.1f}", (480,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Her 3 karede bir küçültülmüş tespitte güncelle
            frame_idx += 1
            small = cv2.resize(raw, (128, 96))
            if frame_idx % 3 == 0:
                small_boxes, small_probs = detector.detect(small)
                if small_boxes is not None and len(small_boxes) > 0:
                    last_boxes = [
                        (int(x1*5),int(y1*5),int(x2*5),int(y2*5))
                        for x1,y1,x2,y2 in small_boxes
                    ]
                    last_probs = small_probs
                else:
                    last_boxes = []
                    last_probs = []

            any_recog = False
            recognized_name = None

            for i, (x1, y1, x2, y2) in enumerate(last_boxes):
                # Güven düşükse atla
                if last_probs[i] < 0.9:
                    continue

                # Koordinat sınırlarını düzelt
                h, w = bgr.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                face_bgr = bgr[y1c:y2c, x1c:x2c]
                # sahte mi gerçek mi?
                if face_bgr.size == 0 or not is_live(face_bgr):
                    cv2.putText(disp, "Spoof!", (x1c, y2c+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    continue

                # gerçek yüzse çerçeve çiz, embedding çıkar ve tanı
                cv2.rectangle(disp, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                face_tensor = get_face_tensor(raw[y1c:y2c, x1c:x2c])
                if face_tensor is None:
                    continue
                emb = resnet(face_tensor.unsqueeze(0))[0].detach()
                sims = [torch.cosine_similarity(emb, e[1], dim=0).item()
                        for e in self.master.embeddings]
                name = "Bilinmeyen"
                if sims and max(sims) > 0.5:
                    name = self.master.embeddings[int(np.argmax(sims))][0]
                    any_recog = True
                    recognized_name = name

                cv2.putText(disp, name, (x1c, y1c - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Tanıma sonucuna göre röle kontrolü
            if any_recog and recognized_name != "Bilinmeyen" and not door_opened:
                # Kişi tanındı, kapıyı aç
                GPIO.output(RELAY_PIN, GPIO.LOW)
                door_opened = True
                self.master.log_action(f"Kapı Açıldı: {recognized_name}")
            elif not any_recog and door_opened:
                # Hiç kimse tanınmıyorsa kapıyı kapat
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                door_opened = False
                self.master.log_action("Kapı Kapandı")

    
                                        
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        cv2.destroyAllWindows()
        self.master.show_frame(MainMenu)

class LogMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Loglar", font=("Arial", 32, "bold"))\
          .place(relx=0.5, y=40, anchor="center")

        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center", width=1000, height=500)
        scrollbar = tk.Scrollbar(container, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.txt = tk.Text(container, font=("Courier", 14), wrap="none", yscrollcommand=scrollbar.set)
        self.txt.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.txt.yview)

        # Ekranın altından biraz daha yukarıda:
        ttk.Button(self, text="← Geri Dön",
                   command=lambda: master.show_frame(MainMenu))\
           .place(relx=0.5, y=650, anchor="center", width=200, height=50)


    def refresh(self, logs):
        """Arayüzü güncellemek için çağır: logs listesini alıp yazdırır."""
        self.txt.delete("1.0", tk.END)
        for line in logs:
            self.txt.insert(tk.END, line + "\n")
        # otomatik en alta kaydır
        self.txt.see(tk.END)


if __name__ == "__main__":
    App().mainloop()
    
    