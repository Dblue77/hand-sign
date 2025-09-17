import cv2
import mediapipe as mp
import json
from gtts import gTTS
import pygame
import os

# Baca teks perkenalan dari file JSON
with open("intro_config.json", "r", encoding="utf-8") as f:
    intro_texts = json.load(f)

# Inisialisasi Mediapipe untuk deteksi tangan
mp_tangan = mp.solutions.hands
mp_gambar = mp.solutions.drawing_utils
deteksi_tangan = mp_tangan.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Buka kamera
kamera = cv2.VideoCapture(0)

# Inisialisasi pygame untuk audio
pygame.mixer.init()

# Siapkan file TTS untuk tiap teks di JSON
tts_files = {}
for kunci, teks in intro_texts.items():
    nama_file = f"{kunci}.mp3"
    if not os.path.exists(nama_file):   
        gTTS(text=teks, lang="id").save(nama_file)
    tts_files[kunci] = nama_file

def putar_tts(kunci):
    """Putar suara dari file mp3"""
    if kunci in tts_files:
        pygame.mixer.music.load(tts_files[kunci])
        pygame.mixer.music.play()

def jari_angkat(landmark, ujung, sendi):
    """Cek apakah jari terangkat (ujung lebih tinggi dari sendi)"""
    return landmark[ujung].y < landmark[sendi].y - 0.02

gerakan_terakhir = ""  

while kamera.isOpened():
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    frame = cv2.flip(frame, 1)  # Cermin agar natural
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = deteksi_tangan.process(rgb)

    teks = "Tunjukkan gesture tangan..."
    kunci = None

    if hasil.multi_hand_landmarks:
        for titik_tangan in hasil.multi_hand_landmarks:
            mp_gambar.draw_landmarks(frame, titik_tangan, mp_tangan.HAND_CONNECTIONS)

            lm = titik_tangan.landmark

            ibu_jari   = jari_angkat(lm, 4, 3)
            telunjuk   = jari_angkat(lm, 8, 6)
            jari_tengah= jari_angkat(lm, 12, 10)
            jari_manis = jari_angkat(lm, 16, 14)
            kelingking = jari_angkat(lm, 20, 18)

            # Deteksi gesture → mapping ke JSON
            if ibu_jari and not telunjuk and not jari_tengah and not kelingking:
                kunci = "OPEN_PALM"
            elif telunjuk and not jari_tengah and not ibu_jari and not kelingking:
                kunci = "THUMBS_UP"
            elif telunjuk and jari_tengah and not ibu_jari and not kelingking:
                kunci = "VICTORY"
            elif kelingking and not telunjuk and not jari_tengah and not ibu_jari:
                kunci = "THANKS"
            else:
                kunci = "LOVE"

            teks = intro_texts[kunci]

    # Putar suara hanya jika gesture berubah
    if kunci and kunci != gerakan_terakhir:
        putar_tts(kunci)
        gerakan_terakhir = kunci

    # Tampilkan teks di layar
    cv2.putText(frame, teks, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)

    cv2.imshow("Pengenalan Diri dengan Gesture Tangan", frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

kamera.release()
cv2.destroyAllWindows()
