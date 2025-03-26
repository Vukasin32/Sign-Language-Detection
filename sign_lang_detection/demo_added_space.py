import pickle
import cv2
import mediapipe as mp
import numpy as np


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    cv2.putText(frame, f'Ready to learn sign language?', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 100), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f'Use your right hand. For start ---> Press s', (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 0, 100), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f'To delete letter ---> Press d', (40, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f'To delete whole message ---> Press c', (40, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50, 0, 255), 2,
                cv2.LINE_AA)
    cv2.imshow('START WINDOW', frame)

    if cv2.waitKey(25) & 0xFF == ord('s'):
        cv2.destroyWindow('START WINDOW')
        break


model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands # This object is used to detect hands
mp_drawing = mp.solutions.drawing_utils # This object is used to draw landmarks on image
mp_drawing_styles = mp.solutions.drawing_styles # This object is used to set style of landmarks

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# dict_letters represents which label corresponds to which letter in original images from data directory
dict_letters = {0: 'C', 1: 'D', 2: 'U', 3: 'L', 4: 'O', 5: 'A', 6: 'E', 7: 'S', 8: 'T', 9: 'M', 10: 'N', 11: 'I', 12: 'R', 13: 'V', 14: 'K', 15: 'P', 16: 'Z', 17: 'J', 18:'B', 19:'G', 20:'F'}

# List container will keep the track of last 20 predictions, if all of 20 predictions are same then the letter will be displayed
# String message will keep message that is displayed on screen
container = []
message = ''
while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_acc = []
    x_ = [] # List of landmarks x coord. of current prediction
    y_ = [] # List of landmarks y coord. of current prediction
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 1: # One hand regime
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_acc.append(x)
            data_acc.append(y)
            x_.append(x)
            y_.append(y)

        if len(data_acc) == 42: # Prediction will be made only if all landmarks can be registered in image
            x1 = int(min(x_) * w) - 10
            x2 = int(max(x_) * w) + 10
            y1 = int(min(y_) * h) - 10
            y2 = int(max(y_) * h) + 10

            label = model.predict([np.array(data_acc)])
            prediction = dict_letters[int(label[0])]
            print(prediction)

            container.append(prediction)
            if len(container) == 20 and len(set(container)) == 1:
                message += prediction
                container = []
            elif len(container) == 20:
                container = []

            cv2.putText(frame, 'Predicted letter is: ' + prediction, (x1 - 15,y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 50, 205), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, 'Current message is: ' + message, (15, h-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 255), 3,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 0), 3)

    elif result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2: # Two hands regime - Space regime
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        print('You are in Space regime (2 hands)')

        container.append(' ')
        if len(container) == 20 and len(set(container)) == 1:
            message += ' '
            container = []
        elif len(container) == 20:
            container = []

        cv2.putText(frame, 'Message looks like: ' + message + '_', (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Space regime is activated!!!' , (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 150, 255), 3,
                    cv2.LINE_AA)
    cv2.imshow('APP WINDOW', frame)

    if cv2.waitKey(5) & 0xFF == ord('d'): # pressing d on keyboard deletes one letter from message
        message = message[:-1]
    if cv2.waitKey(5) & 0xFF == ord('c'): # pressing c on keyboard deletes complete message
        message = ''
    if cv2.waitKey(5) & 0xFF == ord('q'): # pressing q on keyboard breaks out from process
        break

cap.release()
cv2.destroyAllWindows()
print('SUCCESS - your message in sign language is: ' + message)

