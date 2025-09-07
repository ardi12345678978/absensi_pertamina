import os
import shutil
import pickle
import face_recognition

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"


def hapus_dataset(nama):
    """Hapus folder dataset milik seorang karyawan"""
    folder = os.path.join(DATASET_DIR, nama)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"[INFO] Dataset untuk '{nama}' berhasil dihapus.")
        rebuild_encodings()
    else:
        print(f"[WARNING] Dataset untuk '{nama}' tidak ditemukan.")


def rebuild_encodings():
    """Bangun ulang file encodings.pkl dari dataset"""
    encodings = []
    names = []
    total_foto = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):  # support jpg/jpeg/png
                path = os.path.join(root, file)
                img = face_recognition.load_image_file(path)
                face_enc = face_recognition.face_encodings(img)

                if face_enc:  # jika wajah terdeteksi
                    encodings.append(face_enc[0])
                    names.append(os.path.basename(root))
                    total_foto += 1

    # Simpan ke pickle
    data = {"encodings": encodings, "names": names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Encoding berhasil diperbarui!")
    print(f"       Total orang : {len(set(names))}")
    print(f"       Total foto  : {total_foto}")
    print(f"       Total encoding : {len(encodings)}")


if __name__ == "__main__":
    nama = input("Masukkan nama karyawan yang ingin dihapus: ").strip()
    if nama:
        hapus_dataset(nama)
    else:
        print("[ERROR] Nama tidak boleh kosong.")
