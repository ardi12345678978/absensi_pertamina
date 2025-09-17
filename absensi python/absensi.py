import cv2
import face_recognition
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict, deque
from datetime import datetime
from time import time
import threading
import pyttsx3

# ====== FUNGSI UNTUK BUAT FILE PER BULAN ======
def get_absensi_filename():
    bulan_file = datetime.now().strftime("%Y-%m")
    return f"absensi_{bulan_file}.xlsx"

ABSENSI_FILE = get_absensi_filename()

ENCODINGS_FILE = "encodings.pkl"

# ====== PARAMETER PENTING YANG BISA DIOPTIMALKAN ======
TOLERANCE = 0.35
UPSAMPLE = 2
COOLDOWN = 20
POPUP_DURATION = 5
DETECTION_SCALE = 0.5
MIN_MATCHES = 2
MARGIN_REQUIRED = 0.12
CONFIRM_WINDOW = 2.0
CONFIRM_THRESHOLD = 2
SPEAK_ENABLED = True
# ======================================================

# load encodings
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

encodings = data.get("encodings", [])

if "metadata" in data:
    metadata = data["metadata"]
else:
    metadata = [{"Nama": nm, "Status": "-", "Fungsi": "-"} for nm in data.get("names", [])]

name_to_encs = defaultdict(list)
name_to_info = {}

for enc, info in zip(encodings, metadata):
    nm = info["Nama"]
    name_to_encs[nm].append(enc)
    name_to_info[nm] = info

unique_names = list(name_to_encs.keys())

last_seen = {}
last_action = None
last_action_time = 0
confirm_times = defaultdict(deque)

# ====== INIT TTS ======
try:
    import platform
    system = platform.system().lower()
    if "darwin" in system:
        tts_engine = pyttsx3.init("nsss")
    elif "windows" in system:
        tts_engine = pyttsx3.init("sapi5")
    else:
        tts_engine = pyttsx3.init("espeak")

    tts_engine.setProperty("rate", 150)
    tts_engine.setProperty("volume", 1.0)
except Exception as e:
    print("[TTS INIT ERROR]", e)
    tts_engine = None


def speak_text(text):
    if not SPEAK_ENABLED or not tts_engine:
        return

    def _speak(t):
        try:
            tts_engine.say(t)
            tts_engine.runAndWait()
        except Exception as e:
            print("[TTS ERROR]", e)

    thr = threading.Thread(target=_speak, args=(text,), daemon=True)
    thr.start()


def recognize_best(face_encoding):
    best_name = "Tidak dikenal"
    best_dist = 1.0
    best_count = 0
    second_best_dist = 1.0

    for nm, enc_list in name_to_encs.items():
        if len(enc_list) == 0:
            continue
        dists = face_recognition.face_distance(enc_list, face_encoding)
        m = float(np.min(dists))
        count_within = int(np.sum(dists <= TOLERANCE))

        if m < best_dist:
            second_best_dist = best_dist
            best_dist = m
            best_name = nm
            best_count = count_within
        elif m < second_best_dist:
            second_best_dist = m

    margin = max(0.0, second_best_dist - best_dist)

    if best_count >= MIN_MATCHES:
        return best_name, best_dist, best_count, margin
    if best_dist <= TOLERANCE and margin >= MARGIN_REQUIRED:
        return best_name, best_dist, best_count, margin

    return "Tidak dikenal", best_dist, best_count, margin


def ensure_absensi_file():
    if not os.path.exists(ABSENSI_FILE):
        df = pd.DataFrame(columns=["Nama", "Status", "Fungsi"])
        df.to_excel(ABSENSI_FILE, index=False)


def load_absensi():
    df = pd.read_excel(ABSENSI_FILE, dtype=str).fillna("")
    if "Nama" not in df.columns:
        df["Nama"] = ""
    if "Status" not in df.columns:
        df["Status"] = "-"
    if "Fungsi" not in df.columns:
        df["Fungsi"] = "-"
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

    info = name_to_info.get(nama, {"Status": "-", "Fungsi": "-"})

    # Pastikan baris untuk nama sudah ada
    if nama not in df["Nama"].values:
        new_row = {"Nama": nama, "Status": info.get("Status", "-"), "Fungsi": info.get("Fungsi", "-")}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Pastikan kolom untuk tanggal hari ini sudah ada
    masuk_col = f"{today} - Jam Masuk"
    pulang_col = f"{today} - Jam Pulang"
    if masuk_col not in df.columns:
        df[masuk_col] = ""
    if pulang_col not in df.columns:
        df[pulang_col] = ""

    idx = df.index[df["Nama"] == nama][0]

    if 6 <= hour_now <= 15:
        if df.at[idx, masuk_col] == "":
            df.at[idx, masuk_col] = time_now
            print(f"[ABSEN MASUK] {nama} ({info['Status']} / {info['Fungsi']}) masuk pada {time_now}")
            last_action = f" {nama} Absen Masuk {time_now}"
            speak_text(f"{nama} sudah absen masuk")
        else:
            print(f"[INFO] {nama} sudah absen masuk hari ini.")
            last_action = f" {nama} sudah absen masuk"
            speak_text(f"{nama} sudah absen masuk hari ini")
    elif 16 <= hour_now <= 23:
        if df.at[idx, masuk_col] != "":
            if df.at[idx, pulang_col] == "":
                df.at[idx, pulang_col] = time_now
                print(f"[ABSEN PULANG] {nama} pulang pada {time_now}")
                last_action = f" {nama} Absen Pulang {time_now}"
                speak_text(f"{nama} sudah absen pulang")
            else:
                print(f"[INFO] {nama} sudah absen pulang hari ini.")
                last_action = f" {nama} sudah absen pulang"
                speak_text(f"{nama} sudah absen pulang hari ini")
        else:
            print(f"[INFO] {nama} belum absen masuk, jadi tidak bisa absen pulang.")
            last_action = f" {nama} belum absen masuk"
            speak_text(f"{nama} belum absen masuk, jadi tidak bisa absen pulang")
    else:
        print(f"[DILUAR WAKTU] {nama} mencoba absen pada {time_now}")
        last_action = f" Diluar jam absen"
        speak_text(f"{nama} mencoba absen diluar jam")

    last_action_time = time()
    save_absensi(df)


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap or not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("[INFO] Sistem absensi berjalan, tekan 'q' untuk keluar.")

DETECT_MODEL = "hog"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations_small = face_recognition.face_locations(
        rgb_small,
        number_of_times_to_upsample=max(1, UPSAMPLE - 1),
        model=DETECT_MODEL
    )

    face_locations = []
    for (top, right, bottom, left) in face_locations_small:
        top = int(top / DETECTION_SCALE)
        right = int(right / DETECTION_SCALE)
        bottom = int(bottom / DETECTION_SCALE)
        left = int(left / DETECTION_SCALE)
        face_locations.append((top, right, bottom, left))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    now_ts = time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name, dist, count, margin = recognize_best(face_encoding)

        if name != "Tidak dikenal":
            dq = confirm_times[name]
            dq.append(now_ts)
            while dq and (now_ts - dq[0]) > CONFIRM_WINDOW:
                dq.popleft()

            if len(dq) >= CONFIRM_THRESHOLD:
                if name not in last_seen or (now_ts - last_seen[name]) > COOLDOWN:
                    update_absensi(name)
                    last_seen[name] = now_ts
                    confirm_times[name].clear()

        color = (0, 255, 0) if name != "Tidak dikenal" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = name if name != "Tidak dikenal" else "Tidak dikenal"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    if last_action and (time() - last_action_time) < POPUP_DURATION:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(last_action, font, scale, thickness)
        x, y = 30, 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 14, y - text_h - 14), (x + text_w + 14, y + 14), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, last_action, (x, y),
                    font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Absensi Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
