import configparser
import os
import sys
import threading
import time
from queue import Queue, Empty, Full
from threading import Thread

import numpy
import numpy as np
import cv2
import face_recognition

import atexit


class ThreadException(Exception):
    pass


class Sentinal(object):
    """
    Object to be shared on the Queue.
    Used to signal to all threads a stop event
    """

    state = 'main'
    """
    Possible values: main, face, capture
    main - When the user hits q in capture worker
    face - Exception in face recognition worker
    capture - Exception in video capture worker
    """


_sentinel = Sentinal()


class NameQueue(Queue):
    def __init__(self):
        super(NameQueue, self).__init__(maxsize=10)


Q = NameQueue()


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
            print(f"Loading known persons {name} -> {image}")
            known_face_names.append(name)
            img = face_recognition.load_image_file(os.path.join('known_persons', image))
            #known_face_encodings.append(face_recognition.face_encodings(img)[0])
            known_face_encodings.append(face_recognition.face_encodings(img))

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

    def run(self):
        print("FaceRec.FrameCaptureWorker started")
        self.prev = 0  # used as a counter for the FPS
        self.dowork = True
        try:
            self._do_work()
        except Exception as e:
            print(f"FaceRec.FaceRecWorker, exception: {e}")
            Q.queue.clear()
            _sentinel.state = 'capture'
            Q.put(_sentinel)
            self.dowork = False

    def _do_work(self):
        while self.dowork:
            try:
                item = Q.get_nowait()
            except Empty as e:
                pass
            else:
                if item is _sentinel:
                    Q.queue.clear()
                    Q.put(_sentinel)
                    self.dowork = False
                else:
                    Q.put(item)

            time_elapsed = time.time() - self.prev

            if time_elapsed > 1. / self.frame_rate:
                self.prev = time.time()
                ret, frame = self.src.read()
                try:
                    Q.put(frame)
                except Full as e:
                    Q.get()
                    Q.put(frame)
                cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("FaceRec.FrameCaptureWorker has to stop, 'q' hit")
                Q.queue.clear()
                _sentinel.state = 'main'
                Q.put(_sentinel)  # we signal the FaceRecWorker that we stopped
                self.dowork = False

        cv2.destroyAllWindows()
        print("FaceRec.FaceCaptureWorker stopped")


class FaceRecWorker(Thread):
    """
    Threaded worker that will search for a face in a opencv frame
    """

    def __init__(self, frameworker, names, encodings):
        super(FaceRecWorker, self).__init__()
        self.frameworker = frameworker
        self.names = names
        self.encodings = encodings
        self.dowork = True
        self.reset()

    def reset(self):
        self.name = ''
        Q.queue.clear()

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
                # print("FaceRec.FaceRecWorker Found name", self.name)
                return self.name

    def run(self):
        print("FaceRec.FaceRecWorker started")
        try:
            self._do_work()
        except Exception as e:
            print(f"FaceRec.FaceRecWorker, exception: {e}")
            Q.queue.clear()
            _sentinel.state = 'face'
            Q.put(_sentinel)
            self.dowork = False

    def _do_work(self):
        while self.dowork:
            try:
                frame = Q.get_nowait()
            except Empty as e:
                continue
            else:
                if frame is _sentinel:
                    Q.queue.clear()
                    Q.put(_sentinel)
                    self.dowork = False

                if frame is not None and isinstance(frame, numpy.ndarray):
                    if self._find_face(frame):
                        Q.put(self.name)
                # else:
                #     print("FaceRec.FaceRecWorker received no frame")
        print("FaceRecWorker stopped")


class Controller(object):
    """
    Control the capture of frames and check for faces
    """

    def __init__(self):
        known = KnownPersons()
        self.names, self.encodings = known.get_data()

        # make sure we always close sources and windows
        atexit.register(self.stop)

    def start(self):
        self.video_capture = cv2.VideoCapture(0)
        # set resolution lower to speeds up FPS
        # Be aware!!
        # Make sure that the webcam supports the resolution that you are setting to using v4l2-ctl command
        # v4l2-ctl --list-formats-ext
        self.video_capture.set(3, 352)  # Setting webcam's image width
        self.video_capture.set(4, 288)  # Setting webcam' image height

        self.dowork = True

        self.frameworker = FrameCaptureWorker(self.video_capture)
        self.frameworker.start()
        time.sleep(0.1)
        self.faceworker = FaceRecWorker(self.frameworker, self.names, self.encodings)
        self.faceworker.start()
        time.sleep(0.1)

        while self.dowork:
            try:
                item = Q.get_nowait()
            except Empty as e:
                pass
            else:
                if item is _sentinel:
                    if _sentinel.state == 'main':
                        print("Capture worker/user wants to stop, stopping all threads")
                        self.stop()
                        self.dowork = None
                    elif _sentinel.state in ('face', 'capture'):
                        raise ThreadException
                elif isinstance(item, str):
                    print(f"Got from queue name: {item}")
                else:
                    Q.put(item)

            time.sleep(0.5)

    def stop(self):
        self.frameworker.dowork = False
        self.frameworker.join()
        self.faceworker.dowork = False
        self.faceworker.join()
        time.sleep(0.1)
        Q.queue.clear()
        self.video_capture.release()
        time.sleep(0.5)


if __name__ == '__main__':
    Contr = Controller()

    try:
        Contr.start()
    except ThreadException as e:
        print(f"Thread exception: {e}")
        Contr.stop()
        print(f"Stopping...")
        time.sleep(1)
        sys.exit(1)





