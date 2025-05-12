# Angela Zhang
# Webcam Posture Detector
# main.py
# Created to detect bad posture using the angles obtained by webcam and MediaPipe Face Mesh 

import time
import cv2
import mediapipe as mp
import numpy as np
from posture import calculate_head_tilt_angles, is_bad_posture
import simpleaudio as sa

# initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)

# initialize webcam
cap = cv2.VideoCapture(0)

# variables
bad_posture_start_time = None
showing_bad_posture = False
sound_played = False
BAD_POSTURE_DELAY = 2  
SIDE_THRESHOLD = 5
UP_THRESHOLD = 10
DOWN_THRESHOLD = 5
DRAW_OVAL = True
TOOLTIP_ON = True

# sound effect for when bad posture is detected
def play_alert():
    sound = sa.WaveObject.from_wave_file("alert.wav")
    sound.play()

# while webcam is opened
while cap.isOpened(): 
    success, image = cap.read()
    if not success:
        break

    # flip the image (i like this way better)
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # if face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            if DRAW_OVAL:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_spec
                )

            # get chin and forehead coordinates based on face landmarks
            chin = face_landmarks.landmark[152]
            forehead = face_landmarks.landmark[10]
            
            # make into points for calculating angles
            chin_point = np.array([chin.x * w, chin.y * h, chin.z])
            forehead_point = np.array([forehead.x * w, forehead.y * h, forehead.z])

            side_angle, up_down_angle = calculate_head_tilt_angles(forehead_point, chin_point, image=image)

            # posture delay
            bad_posture = is_bad_posture(side_angle, up_down_angle, side_threshold=SIDE_THRESHOLD, up_threshold=UP_THRESHOLD, down_threshold=DOWN_THRESHOLD)

            # check if bad posture is detected and mark the time
            current_time = time.time()
            if bad_posture:
                if bad_posture_start_time is None:
                    bad_posture_start_time = current_time  # start timer
                elif current_time - bad_posture_start_time >= BAD_POSTURE_DELAY: # if time is up past the delay time
                    showing_bad_posture = True
                    if BAD_POSTURE_DELAY >= 1 and not sound_played:
                        play_alert()
                        print("Bad Posture Detected! Sound notification played.")
                        sound_played = True

            else:
                bad_posture_start_time = None
                showing_bad_posture = False
                sound_played = False


            # LABEL SECTION
            if showing_bad_posture:
                cv2.putText(image, 'BAD POSTURE', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (50, 50, 255), 9)
            else:
                cv2.putText(image, 'GOOD POSTURE', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 220, 0), 9)

            # show angles
            vertical_direction = "Down" if up_down_angle < 0 else "Up"
            side_direction = "Right" if side_angle < 0 else "Left"
            cv2.putText(image, f'Side Angle: {side_angle:.1f} ({side_direction})', (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            cv2.putText(image, f'Vertical Angle: {up_down_angle:.1f} ({vertical_direction})', (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            
            # show thresholds and delay time and oval status
            cv2.putText(image, f'Delay Time: {BAD_POSTURE_DELAY} seconds', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            oval_status = "ON" if DRAW_OVAL else "OFF"
            oval_color = (0, 0, 255) if DRAW_OVAL else (100, 100, 100)
            cv2.putText(image, f'Oval: {oval_status}', (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, oval_color, 2)

            cv2.putText(image, 'Tilt Thresholds:', (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(image, f'Left/Right: {SIDE_THRESHOLD}', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(image, f'Up: {UP_THRESHOLD}', (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(image, f'Down: {DOWN_THRESHOLD}', (50, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(image, 'Tooltips: G', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # show keys to change parameters if TOOLTIP_ON is True
            if TOOLTIP_ON:
                cv2.putText(image, 'Keys (first one is increase, second is decrease):', (30, h - 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'ESC: Exit', (50, h - 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'Q/A: Side Threshold', (50, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'W/S: Up Threshold', (50, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'E/D: Down Threshold', (50, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'R/F: Delay Time', (50, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(image, 'T: Toggle Oval', (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            

    # show the video image
    cv2.imshow('Webcam Posture Checker', image)
    
    # key bindings for changing parameters and exiting
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('q'):
        SIDE_THRESHOLD += 1
    elif key == ord('a'):
        SIDE_THRESHOLD = max(0, SIDE_THRESHOLD - 1)
    elif key == ord('w'):
        UP_THRESHOLD += 1
    elif key == ord('s'):
        UP_THRESHOLD = max(0, UP_THRESHOLD - 1)
    elif key == ord('e'):
        DOWN_THRESHOLD += 1
    elif key == ord('d'):
        DOWN_THRESHOLD = max(0, DOWN_THRESHOLD - 1)
    elif key == ord('r'):
        BAD_POSTURE_DELAY += 1
    elif key == ord('f'):
        BAD_POSTURE_DELAY = max(0, BAD_POSTURE_DELAY - 1)
    elif key == ord('t'):
        DRAW_OVAL = not DRAW_OVAL
    elif key == ord('g'):
        TOOLTIP_ON = not TOOLTIP_ON


# clean up
cap.release()
face_mesh.close()
cv2.destroyAllWindows()