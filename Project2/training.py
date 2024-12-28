from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import cv2
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#brings in the holistic model to make the detections
mp_holistic =mp.solutions.holistic
face_connections = mp.solutions.face_mesh
#drawing utilities to draw what is detected
mp_drawing = mp.solutions.drawing_utils

actions= np.array(['hello' , 'thanks', 'please'])
num_sequences = 30
sequence_length =30
DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')



label_map = {label:num for num , label in enumerate(actions)}

print(label_map)
actions= np.array(['hello' , 'thanks', 'please'])

num_sequences = 30 
#30 framesq
sequence_length= 30

def mediapipe_detection(image , model):
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image.flags.writeable =False
    results = model.process(image)
    image.flags.writeable = True
    image =cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    return image ,results


def draw_landmarks(image , results):
    mp_drawing.draw_landmarks(image , results.face_landmarks,face_connections.FACEMESH_TESSELATION )
    mp_drawing.draw_landmarks(image , results.pose_landmarks,mp_holistic.POSE_CONNECTIONS )
    mp_drawing.draw_landmarks(image , results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS )
    mp_drawing.draw_landmarks(image , results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS )


def draw_styled_landmarks(image , results):
    mp_drawing.draw_landmarks(image , results.face_landmarks,face_connections.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80 , 110 , 10), thickness=1 , circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80 , 256 , 121), thickness=1 , circle_radius=1)

                            )
    mp_drawing.draw_landmarks(image , results.pose_landmarks,mp_holistic.POSE_CONNECTIONS ,
                            mp_drawing.DrawingSpec(color=(80 , 22 , 10), thickness=2 , circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80 , 44 , 121), thickness=2 , circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image , results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121 , 22 , 76), thickness=2 , circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121 , 44 , 250), thickness=2 , circle_radius=2) )
    mp_drawing.draw_landmarks(image , results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS , 
                            mp_drawing.DrawingSpec(color=(245 , 117 , 66), thickness=2 , circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245 , 66 , 230), thickness=2 , circle_radius=2))



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

sequences , labels = [], []
for action in actions:
    for sequence in range(num_sequences):
        window=[]
        for frame_num in range(sequence_length):
            res=np.load(os.path.join(DATA_PATH, action ,str(sequence) , "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)
# print(np.array(labels).shape)

x =np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)


log_dir = os.path.join('logs')
tb_callback =TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=500, callbacks=[tb_callback])


model.summary()



res = model.predict(x_test)

print(res)

model.save('action.h5')

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))



colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
# 1. New detection variables
sequence = []
sentence = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]
        # sequence.append(keypoints)
        # sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()