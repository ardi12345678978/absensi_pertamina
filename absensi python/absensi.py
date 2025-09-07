import cv2
import face_recognition
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from time import time

ENCODINGS_FILE = "encodings.pkl"
ABSENSI_FILE = "absensi.xlsx"

# ====== PARAMETER PENTING YANG BISA DIOPTIMALKAN ======
TOLERANCE = 0.50         # makin kecil = makin ketat (coba 0.45 jika masih sering salah)
UPSAMPLE = 2             # 1-2; lebih besar bantu wajah kecil/jauh
COOLDOWN = 20            # detik jeda anti-spam
POPUP_DURATION = 5       # detik tampilan popup
# ======================================================

# Baca database wajah
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

encodings = data["encodings"]

# Perbaikan → gunakan metadata, bukan hanya names
if "metadata" in data:
    metadata = data["metadata"]
else:
    # fallback ke versi lama (names saja)
    metadata = [{"Nama": nm, "Status": "-", "Fungsi": "-"} for nm in data["names"]]

# Kelompokkan encoding per-nama agar voting/penilaian per orang, bukan per-encoding
name_to_encs = defaultdict(list)
name_to_info = {}

for enc, info in zip(encodings, metadata):
    nm = info["Nama"]
    name_to_encs[nm].append(enc)
    name_to_info[nm] = info  # simpan metadata (Nama, Status, Fungsi)

unique_names = list(name_to_encs.keys())

# Cooldown anti-spam
last_seen = {}

# State popup
last_action = None
last_action_time = 0

def recognize_best(face_encoding):
    """
    Pilih nama dengan jarak encoding terkecil (min distance) dari seluruh encoding milik nama tsb.
    Hanya return nama jika jarak < TOLERANCE, kalau tidak -> 'Tidak dikenal'.
    """
    best_name = "Tidak dikenal"
    best_dist = 1.0

    for nm, enc_list in name_to_encs.items():
        dists = face_recognition.face_distance(enc_list, face_encoding)
        if len(dists) == 0:
            continue
        m = float(np.min(dists))
        if m < best_dist:
            best_dist = m
            best_name = nm

    if best_dist <= TOLERANCE:
        return best_name, best_dist
    else:
        return "Tidak dikenal", best_dist

def ensure_absensi_file():
    """Pastikan file absensi ada dan kolomnya benar (semua kolom bertipe string)."""
    if not os.path.exists(ABSENSI_FILE):
        df = pd.DataFrame(columns=["Nama", "Status", "Fungsi", "Tanggal", "Jam Masuk", "Jam Pulang"])
        df.to_excel(ABSENSI_FILE, index=False)

def load_absensi():
    """Baca absensi sebagai string agar tidak ada warning dtype saat update."""
    df = pd.read_excel(ABSENSI_FILE, dtype=str).fillna("")
    # Pastikan kolom lengkap
    for col in ["Nama", "Status", "Fungsi", "Tanggal", "Jam Masuk", "Jam Pulang"]:
        if col not in df.columns:
            df[col] = ""
    return df

def save_absensi(df):
    df.to_excel(ABSENSI_FILE, index=False)

def update_absensi(nama):
    global last_action, last_action_time

    ensure_absensi_file()
    df = load_absensi()

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    hour_now = datetime.now().hour

    # Cek apakah orang sudah punya baris hari ini
    row = df[(df["Nama"] == nama) & (df["Tanggal"] == today)]
    info = name_to_info.get(nama, {"Status": "-", "Fungsi": "-"})

    # Aturan waktu absensi
    if 6 <= hour_now <= 15:
        # Hanya absen masuk
        if row.empty:
            new_row = {
                "Nama": nama,
                "Status": info.get("Status", "-"),
                "Fungsi": info.get("Fungsi", "-"),
                "Tanggal": today,
                "Jam Masuk": time_now,
                "Jam Pulang": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"[ABSEN MASUK] {nama} ({info['Status']} / {info['Fungsi']}) masuk pada {time_now}")
            last_action = f"✅ {nama} Absen Masuk {time_now}"
        else:
            print(f"[INFO] {nama} sudah absen masuk hari ini.")
            last_action = f"ℹ️ {nama} sudah absen masuk"

    elif 16 <= hour_now <= 23:
        # Hanya absen pulang
        if not row.empty:
            idx = row.index[0]
            if df.at[idx, "Jam Pulang"] == "":
                df.at[idx, "Jam Pulang"] = time_now
                print(f"[ABSEN PULANG] {nama} pulang pada {time_now}")
                last_action = f"✅ {nama} Absen Pulang {time_now}"
            else:
                print(f"[INFO] {nama} sudah absen pulang hari ini.")
                last_action = f"ℹ️ {nama} sudah absen pulang"
        else:
            print(f"[INFO] {nama} belum absen masuk, jadi tidak bisa absen pulang.")
            last_action = f"ℹ️ {nama} belum absen masuk"

    else:
        # Diluar jam absen
        print(f"[DILUAR WAKTU] {nama} mencoba absen pada {time_now}")
        last_action = f"⏰ Diluar jam absen"

    last_action_time = time()
    save_absensi(df)

# Mulai kamera (prefer AVFoundation di macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap or not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("[INFO] Sistem absensi berjalan, tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah (UPSAMPLE bantu wajah kecil/jauh)
    face_locations = face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample=UPSAMPLE,
        model="hog"
    )
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name, dist = recognize_best(face_encoding)

        # Hanya proses jika nama valid
        if name != "Tidak dikenal":
            now = time()
            if name not in last_seen or (now - last_seen[name]) > COOLDOWN:
                update_absensi(name)
                last_seen[name] = now

        # Kotak + label
        color = (0, 255, 0) if name != "Tidak dikenal" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = name if name != "Tidak dikenal" else "Tidak dikenal"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Popup dengan background semi-transparan (5 detik)
    if last_action and (time() - last_action_time) < POPUP_DURATION:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(last_action, font, scale, thickness)
        x, y = 30, 60

        # Background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 14, y - text_h - 14), (x + text_w + 14, y + 14), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Teks
        cv2.putText(frame, last_action, (x, y),
                    font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Absensi Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
