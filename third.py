import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3)

DATA_DIR = './data'
data  = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results =hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            print(f'Deleted image: {img_path}')
            os.remove(os.path.join(DATA_DIR,dir_,img_path))
        else: 
            for hands_landmarks in results.multi_hand_landmarks:
                for i in range(len(hands_landmarks.landmark)):
                    x = hands_landmarks.landmark[i].x
                    y = hands_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if (len(data_aux) == 42):
                data.append(data_aux)
                labels.append(dir_)
            




f = open('data.pickle' , 'wb')
pickle.dump({'data': data, 'labels':labels}, f)
f.close()