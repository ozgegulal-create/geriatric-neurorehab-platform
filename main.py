import cv2
import mediapipe as mp
import math
import time
import numpy as np

class NeuroRehabPlatform:
    def __init__(self):
        # MediaPipe Hands Başlatma
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Durum ve Mod Değişkenleri
        self.mode = 'IDLE'
        self.start_time = 0
        self.duration = 20

        # Tapping Sayac Mantığı
        self.state = "OPEN"
        self.threshold_open = 90
        self.threshold_close = 30
        self.tap_count = 0
        self.amplitudes = []

        # Biofeedback Parametreleri
        self.target_radius = 50

        # Ölçüm Veri Kayıtları
        self.results = {
            'PRE_TEST': {'taps': 0, 'freq': 0.0, 'fatigue': 0.0, 'max_amp': 0},
            'POST_TEST': {'taps': 0, 'freq': 0.0, 'fatigue': 0.0, 'max_amp': 0}
        }

    def calculate_fatigue_index(self, amplitudes):
        if len(amplitudes) < 5:
            return 0.0
        n = len(amplitudes)
        k = max(1, int(n * 0.2))
        initial_amp = np.mean(amplitudes[:k])
        final_amp = np.mean(amplitudes[-k:])
        if initial_amp == 0:
            return 0.0
        fatigue = ((initial_amp - final_amp) / initial_amp) * 100
        return max(0.0, round(fatigue, 2))

    def reset_session(self, mode_name, duration=20):
        self.mode = mode_name
        self.duration = duration
        self.start_time = time.time()
        self.tap_count = 0
        self.amplitudes = []
        self.state = "OPEN"

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            current_time = time.time()
            elapsed_time = current_time - self.start_time if self.mode != 'IDLE' else 0
            remaining_time = max(0, int(self.duration - elapsed_time))

            # 1. EL VE PARMAK UCU TAKİBİ
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    x1, y1 = int(thumb.x * w), int(thumb.y * h)
                    x2, y2 = int(index.x * w), int(index.y * h)

                    distance = math.hypot(x2 - x1, y2 - y1)

                    cv2.circle(frame, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    if self.mode in ['PRE_TEST', 'EXERCISE', 'POST_TEST']:
                        if distance < self.threshold_close and self.state == "OPEN":
                            self.state = "CLOSED"
                            self.tap_count += 1
                        elif distance > self.threshold_open and self.state == "CLOSED":
                            self.state = "OPEN"
                            self.amplitudes.append(distance)

            # 2. MOD YÖNETİMİ
            if self.mode in ['PRE_TEST', 'EXERCISE', 'POST_TEST']:
                if elapsed_time >= self.duration:
                    freq = round(self.tap_count / self.duration, 2) if self.duration > 0 else 0
                    fatigue = self.calculate_fatigue_index(self.amplitudes)
                    max_amp = int(np.max(self.amplitudes)) if len(self.amplitudes) > 0 else 0

                    if self.mode in ['PRE_TEST', 'POST_TEST']:
                        self.results[self.mode] = {
                            'taps': self.tap_count,
                            'freq': freq,
                            'fatigue': fatigue,
                            'max_amp': max_amp
                        }
                    self.mode = 'IDLE'

            # 3. ARAYÜZ
            cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), cv2.FILLED)
            cv2.putText(frame, f"MOD: {self.mode}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"KALAN SURE: {remaining_time}s", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Geriatrik Dijital NeuroRehab Platformu", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.reset_session('PRE_TEST', duration=20)
            elif key == ord('2'):
                self.reset_session('EXERCISE', duration=30)
            elif key == ord('3'):
                self.reset_session('POST_TEST', duration=20)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = NeuroRehabPlatform()
    app.run()
