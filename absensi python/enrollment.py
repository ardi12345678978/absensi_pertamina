import cv2
import face_recognition
import os
import pickle
import shutil

# Folder dataset wajah
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"


def daftar_wajah(nama, status, fungsi):
    # Path folder karyawan
    folder = os.path.join(DATASET_DIR, nama)

    # Jika sudah ada → hapus dulu folder lamanya
    if os.path.exists(folder):
        print(f"[WARNING] Data untuk {nama} sudah ada, menghapus dataset lama...")
        shutil.rmtree(folder)

    # Buat folder baru
    os.makedirs(folder, exist_ok=True)

    # Simpan metadata ke file info.txt
    with open(os.path.join(folder, "info.txt"), "w") as f:
        f.write(f"Nama: {nama}\n")
        f.write(f"Status: {status}\n")
        f.write(f"Fungsi: {fungsi}\n")

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Ambil gambar wajah untuk {nama}, tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Kalau ada wajah terdeteksi → crop & simpan
        for (top, right, bottom, left) in face_locations:
            face_img = frame[top:bottom, left:right]

            if count < 10:
                filepath = os.path.join(folder, f"img{count}.jpg")
                cv2.imwrite(filepath, face_img)
                print(f"[INFO] Foto {count+1} disimpan: {filepath}")
                count += 1

            # Kotak hijau di layar
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Pendaftaran Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Encode wajah (hanya dari dataset folder)
    encodings = []
    metadata = []  # Simpan dict {nama, status, fungsi}

    for person in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person)
        info_file = os.path.join(person_folder, "info.txt")

        # Skip kalau tidak ada info
        if not os.path.exists(info_file):
            continue

        # Baca metadata
        info = {}
        with open(info_file, "r") as f:
            for line in f:
                key, val = line.strip().split(": ", 1)
                info[key] = val

        for file in os.listdir(person_folder):
            if file.endswith(".jpg"):
                path = os.path.join(person_folder, file)
                img = face_recognition.load_image_file(path)
                face_enc = face_recognition.face_encodings(img)
                if face_enc:
                    encodings.append(face_enc[0])
                    metadata.append(info)

    # Simpan ke file pickle
    data = {"encodings": encodings, "metadata": metadata}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Pendaftaran selesai untuk {nama}!")


if __name__ == "__main__":
    nama = input("Masukkan nama karyawan: ").strip()
    status = input("Masukkan status (contoh: Karyawan/Magang): ").strip()
    fungsi = input("Masukkan fungsi/departemen (contoh: IT/HC/Finance): ").strip()

    if not nama or not status or not fungsi:
        print("[ERROR] Nama, status, dan fungsi tidak boleh kosong!")
    else:
        daftar_wajah(nama, status, fungsi)
