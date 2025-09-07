import os
import pickle
import face_recognition

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"

def rebuild_encodings():
    encodings = []
    metadata = []

    for person in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person)
        info_file = os.path.join(person_folder, "info.txt")

        if not os.path.isdir(person_folder) or not os.path.exists(info_file):
            continue

        # Baca metadata (Nama, Status, Fungsi)
        info = {}
        with open(info_file, "r") as f:
            for line in f:
                if ": " in line:
                    key, val = line.strip().split(": ", 1)
                    info[key] = val

        # Cari semua gambar
        for file in os.listdir(person_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(person_folder, file)
                img = face_recognition.load_image_file(path)
                face_enc = face_recognition.face_encodings(img)

                if face_enc:
                    encodings.append(face_enc[0])
                    metadata.append(info)

    data = {"encodings": encodings, "metadata": metadata}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Encoding berhasil dibuat ulang!")
    print(f"       Total orang: {len(set(m['Nama'] for m in metadata))}")
    print(f"       Total foto : {len(metadata)}")


if __name__ == "__main__":
    rebuild_encodings()
