import time

import numpy as np
import cv2
import face_recognition

image_0 = face_recognition.load_image_file('KnownImages/marek_0.png')
image_1 = face_recognition.load_image_file('KnownImages/stas_0.png')

face_enc_0 = face_recognition.face_encodings(image_0)[0]
face_enc_1 = face_recognition.face_encodings(image_1)[0]

known_face_encodings = [face_enc_0, face_enc_1]
known_face_names = ['Marek', 'Stas']

video_capture = cv2.VideoCapture(0)
# set resolution lower to speeds up FPS
# Be aware!!
# Make sure that the webcam supports the resolution that you are setting to using v4l2-ctl command
# v4l2-ctl --list-formats-ext
video_capture.set(3, 352) #Setting webcam's image width
video_capture.set(4, 288)  # Setting webcam' image height

while True:

    ret, frame = video_capture.read()

    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Converting the frame from OpenCV's BGR format to the RGB format
    rgb_frame = small_frame[:, :, ::-1]

    # Finding the face locations and encodings in each frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Now to loop through each face in this frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Checking if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = 'Unknown'

        # Use the known face with the smallest vector distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back up face locations since the frame we detected in was scaled to 1/10 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

