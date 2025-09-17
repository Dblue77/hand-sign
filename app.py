import cv2
import mediapipe as mp
import json
from gtts import gTTS
import pygame
import os

with open("intro_config.json", "r", encoding="utf-8") as f:
    intro_texts = json.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

pygame.mixer.init()

tts_files = {}
for key, val in intro_texts.items():
    filename = f"{key}.mp3"
    if not os.path.exists(filename):   
        gTTS(text=val, lang="id").save(filename)
    tts_files[key] = filename

def play_tts(key):
    """Putar suara dari file cache"""
    if key in tts_files:
        pygame.mixer.music.load(tts_files[key])
        pygame.mixer.music.play()

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y - 0.02

last_key = ""  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    text = "Tunjukkan gesture tangan..."
    key = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            thumb_up  = finger_up(lm, 4, 3)
            index_up  = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)
            ring_up   = finger_up(lm, 16, 14)
            pinky_up  = finger_up(lm, 20, 18)

            if thumb_up and not index_up and not middle_up and not pinky_up:
                key = "OPEN_PALM"
            elif index_up and not middle_up and not thumb_up and not pinky_up:
                key = "THUMBS_UP"
            elif index_up and middle_up and not thumb_up and not pinky_up:
                key = "VICTORY"
            elif pinky_up and not index_up and not middle_up and not thumb_up:
                key = "THANKS"
            else:
                key = "LOVE"

            text = intro_texts[key]

    # === mainkan suara hanya kalau gesture berubah ===
    if key and key != last_key:
        play_tts(key)
        last_key = key

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)

    cv2.imshow("Hand Sign Perkenalan Diri", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC keluar
        break

cap.release()
cv2.destroyAllWindows()
