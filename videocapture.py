'''
  Video Capture thread
'''
import cv2
import threading
import time

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

outputFrame = None
lock = threading.Lock()

# Define video capture class
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, driver=None):
        self.src = src
        logger.info('[i] Opening media device...')
        if driver is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, driver)

        time.sleep(2)
        self.started = False
        self.thread = None
        if self.cap.isOpened():
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.grabbed, self.frame = self.cap.read()
            self.read_lock = threading.Lock()
        else:
            logger.error('RTSP open failed')

    def get(self, var1):
        return self.cap.get(var1)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            logger.info('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        global outputFrame, lock
        while self.started:
            grabbed, frame = self.cap.read()
            #time.sleep(0.01)
            if frame is None:
                return
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                #with lock:
                    #outputFrame = frame.copy()

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
