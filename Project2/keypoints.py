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


DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')
actions= np.array(['hello' , 'thanks' , 'please'])

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



cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        #read the camera feed
        ret , frame  = cap.read()

        #make the detections
        image , results = mediapipe_detection(frame , holistic)
        # print(results)

        #draw the landmarks
        draw_styled_landmarks(image , results)


        
        #show to screemq
        cv2.imshow("OpenCV Feed", image )

        if cv2.waitKey(10) & 0xFF ==  ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


result_test = extract_keypoints(results)
print(result_test)

np.save("0" , result_test)




for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action ,str(sequence)))
        except: 
            break



cap= cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(num_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()

