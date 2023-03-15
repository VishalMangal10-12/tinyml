#!/usr/bin/env python
'''
  Main application
'''
import cv2
import argparse
import numpy as np
import threading
import time
from PIL import Image
import requests

## Import VideoCapture
from videocapture import VideoCaptureAsync, outputFrame, lock

## Import CHIPDetection Model implementation
from modelchipdetection import CHIPDetection
import serial
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logging.disable(logging.DEBUG)
#logging.disable(logging.INFO)
logger.disabled = True



import sys ## ROBOTICARM
#sys.path.append('./uarm/uArm-Python-SDK-2.0')  # path of uArm-Python-SDK-2.0 ## ROBOTICARM
#from uarm.wrapper import SwiftAPI ## ROBOTICARM
##Rbotic Arm trigger
trigger = 0


FLAGS = None
outputFrame = None
CB_COM_PORT = 'COM10'
RB_COM_PORT = 'COM5'
lock = threading.Lock()

####################################### Flask Nodes #######################################
from flask import Response
from flask import Flask
from flask import render_template

# initialize a flask object
app = Flask(__name__)

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            #outputFrame = imutils.resize(outputFrame, width=320)
            #outputFrame = cv2.resize(outputFrame, (320,200))
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/reset")
def reset():
    model.badCounter = 0
    return "Ok"


belt_serial = None
    
# swift = SwiftAPI(port=RB_COM_PORT)
# print("Swift")
# print(swift.port)
# print(swift.firmware_version)

## Robotic Arm Sequences
def Bad_PCB_Posn():
    #global swift
    print("BAd Image")
    belt_move(False)
    # swift.reset()
    # swift.set_speed_factor(0.2)
    # swift.set_position(x=160, y=-3, z=38, speed=100000)
    # #swift.set_position(relative = True, z = -77)
    # swift.set_pump(on=True)
    # swift.set_gripper(catch=True)
    # swift.set_position(relative = True, z = 100)
    # swift.set_position(x=283, y=-184, z=82, speed=100000)
    # swift.set_position(relative = True, z = -40)
    # time.sleep(1)
    # swift.set_pump(on=False)
    # swift.set_gripper(catch=False)
    # swift.reset()
    # logger.info("Starting Belt")
    # belt_move(True)
    #writeSerial = False
           
def Good_PCB_Posn():
    #global swift
    logger.debug("Picking Good")
    belt_move(True)
    # try:
    #     swift.reset()
    #     swift.set_speed_factor(0.2)
    #     swift.set_position(x=160, y=-3, z=38, speed=100000)
    #     #swift.set_position(relative = True, z = -77)
    #     swift.set_pump(on=True)
    #     swift.set_gripper(catch=True)
    #     swift.set_position(relative = True, z = 100)
    #     swift.set_position(x=175, y=-227, z=100, speed=100000)
    #     #swift.set_position(relative = True, z = -40)
    #     time.sleep(1)
    #     swift.set_pump(on=False)
    #     swift.set_gripper(catch=False)
    #     swift.set_position(relative = True, z = 40)
    #     swift.reset()
    # except:
    #     logger.error('[!] Swift...', exc_info = True)
    # logger.debug("Placed Good")

def ManualCheck_PCB_Posn():
    #global swift
    logger.debug("Picking Manual Check")
    #swift.reset()
    #swift.set_speed_factor(0.2)
    swift.set_position(x=104, y=214, z=79)
    swift.set_position(relative = True, z = -77)
    swift.set_pump(on=True)
    swift.set_gripper(catch=True)
    swift.set_position(relative = True, z = 79)
    swift.set_position(x=175, y=-227, z=79)
    swift.set_position(relative = True, z = -40)
    time.sleep(1)
    swift.set_pump(on=False)
    swift.set_gripper(catch=False)
    #swift.reset()
    swift.set_position(x=200, y=0, z=150)
    logger.debug("Placed Manual Check")

def belt_init():
    global belt_serial
    belt_serial = serial.Serial()  # open serial port
    belt_serial.baudrate = 115200
    belt_serial.port = CB_COM_PORT
    belt_serial.writeTimeout = .3
    belt_serial.open()
    if belt_serial.is_open:
        print("Belt Serial Port open {}".format(belt_serial.name))
    else:
        print("Could not open Belt Serial")       # check which port was really used

def belt_move(moveStatus):
    print(f"method called {moveStatus}")
    global belt_serial
    # if moveStatus == True:
    #     url = "http://127.0.0.1:5000/start"
    #     response = requests.get(url)  
    # else:
    #     url = "http://127.0.0.1:5000/stop"
    #     response = requests.get(url)   
    if belt_serial is not None:
        if moveStatus == True:
            belt_serial.write(b'M')
            print("Belt Move write")
        else :
            belt_serial.write(b'S')
            print("Belt stop write")
        belt_serial.flush()
    else :
        print ("ERROR: Belt not open/initialized!")

def roboticarm_thread():
    global trigger
    while True:
        if trigger != 0:
            ## Robotic arm movement
            ## Robotic movement
            if trigger == 2:
                Bad_PCB_Posn()
            elif trigger == 1:
                Good_PCB_Posn()
            #elif trigger == 3:
                #ManualCheck_PCB_Posn()
            trigger = 0
            #writeSerial = False
####################################### Model Pipeline #######################################
def model_pipeline():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, capture, trigger
    
    if FLAGS.model_name == 'chipdetection':
        model = CHIPDetection(threshold=FLAGS.model_threshold, tritonurl=FLAGS.triton_url, samplesize=FLAGS.sample_size, samplethreshold=FLAGS.sample_threshold)
    else:
        logger.error("Model not supported")
        return

    framecount = 0
    #start video capture
    logger.info('[i] Starting Video Capture pipeline...')
    capture.start()
    ret, frame = capture.read()
    print('#################################',frame)
    if ret:
        x1 = int(frame.shape[1] * 0.5)
        y1 = int(frame.shape[0] * 0.2)
        #x2 = x1 + int(frame.shape[1] * 0.8) #x1+480 #int(frame.shape[1] * 0.8)
        #y2 = y1 + int(frame.shape[1] * 0.8) #(y1+480) #int(frame.shape[0] * 0.8)
        x2 = int(frame.shape[1] * 0.7)
        y2 = int(frame.shape[0] * 0.8)
        a1 = int(frame.shape[1] * 0.2)
        b1 = int(frame.shape[0] * 0.2)
        a2 = int(frame.shape[1] * 0.8)
        b2 = int(frame.shape[0] * 0.8)
    while True:
        ret, frame = capture.read()
        in_image = frame[y1:y2,x1:x2]
        if ret == False:
            logger.warning('[!] Exiting the pipeline due to no frame available or error in reading the frame')
            break
        framecount += 1
        try:
            # HAND Detection using Color Threshold
            org_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            min_HSV = np.array([80, 216, 164], dtype="uint8")
            max_HSV = np.array([120, 216, 164], dtype="uint8")
            image = np.asarray(org_image.convert('HSV'))
            skinRegionHSV = cv2.inRange(image, min_HSV, max_HSV)
            skinHSV = cv2.bitwise_and(image, image, mask=skinRegionHSV)
            average = skinHSV.mean(axis=0).mean(axis=0)
            #print(f"Hand {average[2]}")
            cv2.rectangle(frame, (a1, b1), (a2, b2), (0, 255, 255), 4)
            if average[2] > 15.2:
                ## Render it
                with lock:
                    cv2.putText(frame, "!!  HAND !!", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 4)
                    outputFrame = frame.copy()
                continue
            # Model Prediction
            start_time = time.time()
            prediction, trigger = model.predict(in_image, frame)
            if prediction:
                if trigger == 2:
                    belt_move(False)
                #     writeSerial = True
            else:
                break
            logger.debug("Model {} Inference time {:.2f} msecs".format(FLAGS.model_name, (time.time()-start_time)*1000))
        except:
            logger.error('[!] Exception during inferencing...', exc_info = True)
            break

        ## Render it
        with lock:
            outputFrame = frame.copy()

    capture.stop()


####################################### Argument Parser #######################################
parser = argparse.ArgumentParser()
parser.add_argument('-s',
                    '--stream-url',
                    type=str,
                    required=False,
                    default='0',
                    help='RTSP Stream URL')
parser.add_argument('-m',
                    '--model-name',
                    type=str,
                    required=False,
                    default='chipdetection',
                    help='Name of model')
parser.add_argument('-op',
                    '--output-port',
                    type=int,
                    required=False,
                    default=8080,
                    help='Frame rendering service PORT. Default is 8080.')
parser.add_argument('-tu',
                    '--triton-url',
                    type=str,
                    required=False,
                    default='localhost:8001',
                    help='Triton server url. Default localhost:8001.')
parser.add_argument('-mt',
                    '--model-threshold',
                    type=float,
                    required=False,
                    default=0.5,
                    help='Model inference output threshold')
parser.add_argument('-ss',
                    '--sample-size',
                    type=int,
                    required=False,
                    default=16,
                    help='Sample Event Window Size')
parser.add_argument('-st',
                    '--sample-threshold',
                    type=int,
                    required=False,
                    default=23,
                    help='Event Threshold in the Event Window')
FLAGS = parser.parse_args()

belt_init()



#def init_video():
    ####################################### Video Capture Thread #######################################
if FLAGS.stream_url == '0':
    capture = VideoCaptureAsync(src=0)
elif FLAGS.stream_url == '1':
    capture = VideoCaptureAsync(src=1)
elif FLAGS.stream_url == '2':
    capture = VideoCaptureAsync(src=2)
elif FLAGS.stream_url == 'F':
    VIDEO_INPUT = "rtsp://admin:xmsx1234@192.168.1.100:554/main"
    capture = VideoCaptureAsync(src=VIDEO_INPUT)
else:
    capture = VideoCaptureAsync(src="rtspsrc location={}".format(str(FLAGS.stream_url)) + " ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", driver=cv2.CAP_GSTREAMER)
   

####################################### Pipeline Thread #######################################
pipeline = threading.Thread(target=model_pipeline, args=())
pipeline.daemon = True
pipeline.start()

# robot = threading.Thread(target=roboticarm_thread, args=())
# robot.daemon = True
# robot.start()

if __name__ == "__main__":
    logger.info('Starting rendering service....')
    app.run(host='0.0.0.0', port=FLAGS.output_port, debug=True, threaded=True, use_reloader=False)
    print("**Main**")
    #init_video()
    pipeline.join()
