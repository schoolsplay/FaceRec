import configparser
import os
import sys
import time
from queue import Queue
from threading import Thread

import numpy as np
import cv2
import face_recognition

import atexit

class NameQueue(Queue):
    def __init__(self):
        super(NameQueue, self).__init__(maxsize=1)


class KnownPersons(object):
    """
    Object to load and store our known persons.
    We get the info from a simple python configparser file that should be up updated by hand.
    See persons.cfg inside known_persons.py
    """

    known_face_encodings = []
    """ Will hold all the known face encodings. Will be in sync with known_face_names"""

    known_face_names = []
    """ Will hold all the known names. Will be in sync with known_face_encodings"""

    conf = configparser.ConfigParser()
    conf.read(os.path.join('known_persons', 'persons.ini'))
    for each_section in conf.sections():
        for name, image in conf.items(each_section):
            known_face_names.append(name)
            known_face_encodings.append(face_recognition.load_image_file(os.path.join('known_persons', image)))

    @staticmethod
    def get_data():
        return KnownPersons.known_face_names, KnownPersons.known_face_encodings


class FrameCaptureWorker(Thread):
    """
    Threaded worker which capture opencv frames from a source
    """
    frame_rate = 10
    """FPS value. We use this to lower the rate of which we read a frame as we can't
    really lower the framerate of a cam"""

    def __init__(self, src):
        """
        :param src: A connected opencv capture object that has a read method
        which should return a frame
        """
        super(FrameCaptureWorker, self).__init__()
        self.src = src
        self.prev = 0  # used as a counter for the FPS
        self.dowork = True

    def run(self):
        while self.dowork:
            time_elapsed = time.time() - self.prev
            ret, self.frame = self.src.read()
            if not ret:
                print("FaceRec.FrameCaptureWorker has to stop, capture read failed")
                self.frame = None
                self.dowork = False

            if time_elapsed > 1. / self.frame_rate:
                self.prev = time.time()
                # Display the image
                cv2.imshow('Video', self.frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("FaceRec.FrameCaptureWorker has to stop, 'q' hit")
                self.frame = None  # we signal the FaceRecWorker that we stopped
                self.dowork = False


class FaceRecWorker(Thread):
    """
    Threaded worker that will search for a face in a opencv frame
    """
    def __init__(self, frameworker, names, encodings, namequeue):
        super(FaceRecWorker, self).__init__()
        self.frameworker = frameworker
        self.names = names
        self.encodings = encodings
        self.namequeue = namequeue
        self.dowork = True
        self.reset()

    def reset(self):
        self.name = ''
        self.namequeue.empty()

    def _find_face(self, frame):
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
            matches = face_recognition.compare_faces(self.encodings, face_encoding)

            # Use the known face with the smallest vector distance to the new face
            face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                self.name = self.names[best_match_index]

                # # Draw a box around the face
                # cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                #
                # # Draw a label with the name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                return self.name

    def run(self):
        while self.dowork:
            frame = self.frameworker.frame
            if frame:
                if self._find_face(frame):
                    self.namequeue.put(self.name)
            else:
                print("FaceRec.FaceRecWorker stopped, got no frame")
                self.dowork = False


class Controller(object):
    """
    Control the capture of frames and check for faces
    """
    def __init__(self):
        known = KnownPersons()
        self.names, self.encodings = known.get_data()

        self.video_capture = cv2.VideoCapture(0)

        # make sure we always close sources and windows
        atexit.register(self.video_capture.release)
        atexit.register(cv2.destroyAllWindows)

        # set resolution lower to speeds up FPS
        # Be aware!!
        # Make sure that the webcam supports the resolution that you are setting to using v4l2-ctl command
        # v4l2-ctl --list-formats-ext
        self.video_capture.set(3, 352)  # Setting webcam's image width
        self.video_capture.set(4, 288)  # Setting webcam' image height

        self.dowork = True

    def start(self):
        while self.dowork:
            self.namequeue = NameQueue()
            self.frameworker = FrameCaptureWorker(self.video_capture).start()
            time.sleep(0.1)
            self.faceworker = FaceRecWorker(self.frameworker, self.names, self.encodings, self.namequeue).start()
            time.sleep(0.1)

            name = self.namequeue.get_nowait()
            if name:
                print(f"Got from queue name: {name}")
            time.sleep(1)


Contr = Controller()
Contr.start()


