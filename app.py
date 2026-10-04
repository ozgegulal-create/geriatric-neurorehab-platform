import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import math
import numpy as np
import av

st.set_page_config(page_title="Dijital Nörorehab", layout="centered")
st.title("🧠 Geriatrik Dijital Nörorehabilitasyon")
st.caption("Telefon kameranızı kullanarak parmak açma-kapama (Finger Tapping) analizini başlatın.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class MobileNeuroProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                distance = math.hypot(x2 - x1, y2 - y1)
                
                # Parmak uçlarına hedef noktalar çiz
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

                # Anlık Mesafe Metni
                cv2.putText(img, f"Mesafe: {int(distance)}px", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="neuro-rehab-mobile",
    video_processor_factory=MobileNeuroProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
