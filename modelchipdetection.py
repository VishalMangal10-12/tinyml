'''
  PPE Model Implementation
'''
import cv2
import numpy as np
import time
from itertools import groupby

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing.image import img_to_array

from modelbase import ModelBase

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.disabled = True

#import sys ## ROBOTICARM
#sys.path.append('./uarm/uArm-Python-SDK-2.0')  # path of uArm-Python-SDK-2.0 ## ROBOTICARM
#from uarm.wrapper import SwiftAPI ## ROBOTICARM

NONE = 0
GOOD = 1
BAD = 2
UNKNOWN = 3

COLOUR_NOCOLOUR = 0
COLOUR_RED = 1
COLOUR_DARKBLUE = 2
COLOUR_LIGHTBLUE = 3

# chip detection model
class CHIPDetection(ModelBase):
    def __init__(self, modelname='chipdetection', threshold=0.5, tritonurl='localhost:8001', samplesize=7, samplethreshold=5, noarm=False):
        super(CHIPDetection, self).__init__(modelname, threshold, tritonurl)
        self.noarm = noarm
        #if self.noarm == False:
            #self.swift = SwiftAPI() ## ROBOTICARM
            #time.sleep(1) ## ROBOTICARM
            #self.swift.reset() ## ROBOTICARM
            #time.sleep(1)
            #self.swift.set_speed_factor(50)
        self.sample = []
        self.sample_size = samplesize
        self.sample_threshold = samplethreshold
        self.consume = True
        self.ccounter = 0
        self.badCounter = 0
        self.goodCounter = 0
        #self.new_model = tf.keras.models.load_model('claexport')

      

    def __preprocess(self):
        #logger.info("Preprocessing...")
        resized_img = cv2.resize(self.image, (224,224))
        resized = np.asarray(resized_img, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
        resized = resized.transpose((0,1,4,2,3))
        resized = resized.reshape(1,3,224,224)
        self.transimage = resized.astype(np.float32)

    def __postprocess(self):
        logger.info("Postprocessing...")
        trigger = 0
        if self.is_tabs():
            colour = self.__detect_color()
            trigger = self.processsample()
            if trigger > 0:
                print(trigger)
            #time.sleep(1)
            if colour == 2:
                if self.noarm == False:
                    self.__addevent(BAD)
                    
            elif colour == 1:
                if self.noarm == False:
                    self.__addevent(GOOD)
                    
            else:
                self.__addevent(NONE)


            if trigger == 2:
                print("bad")
                self.badCounter = self.badCounter + 1
                #cv2.putText(self.frame, "BAD", (20, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 4)
            elif trigger == 1:
                print("good")
                self.goodCounter = self.goodCounter + 1
                #cv2.putText(self.frame, "GOOD", (20, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)

            msg = f"Good Count: {self.goodCounter}  Bad Count: {self.badCounter}"
            msg = f"Defective Count: {self.badCounter}"
            cv2.putText(self.frame, msg, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)

        return trigger
        
    def predict(self, image, frame):      
        self.image = image
        self.frame = frame
        self.__preprocess()
        trigger = self.__postprocess()
        
        image = self.image
        frame = self.frame
        return True, trigger

    def is_tabs(self):
        #pre = self.new_model.predict(self.img_array)
        #pre.flatten()
        return True

    def processsample(self):
        if len(self.sample) >= self.sample_size:
            o, n = self.__occurrence()
            self.sample.clear()
            if n >= self.sample_threshold:
                logger.info(self.__eventname(o))
                '''if o == BAD:
                    if self.noarm == False:
                        self.__Bad_PCB_Posn() ## ROBOTICARM
                elif o == GOOD:
                    if self.noarm == False:
                        self.__Good_PCB_Posn() ## ROBOTICARM'''
                return o
        return NONE

    def __occurrence(self):
        occurrence, num_times = 0, 0
        for key, values in groupby(self.sample, lambda x : x):
            val = len(list(values))
            if val >= num_times:
                occurrence, num_times =  key, val
        return occurrence, num_times
    
    def __addevent(self, event):
        self.sample.append(event)

    def __eventname(self, event):
        if event == GOOD:
            return ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[GOOD]"
        if event == BAD:
            return ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[BAD]"
        else:
            return ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[NO]"

    def __detect_color(self):

        code = self.__detect_code(COLOUR_LIGHTBLUE)
        if code == 1:
            return BAD
        elif code == 2:
            return GOOD
        else:
            ## Color code for Unknown
            return COLOUR_NOCOLOUR
        # if self.__detect_code(COLOUR_RED) == True:
        #     print("RED")
        #     return COLOUR_RED
        # if self.__detect_code(COLOUR_DARKBLUE) == 1:
        #     print("ORANGE")
        #     return BAD
        # if self.__detect_code(COLOUR_DARKBLUE) == 2:
        #     print("ORANGE")
        #     return GOOD
        # if self.__detect_code(COLOUR_LIGHTBLUE) == True:
        #     print("BLUE")
        #     return COLOUR_LIGHTBLUE
        # else:
        #     ## Color code for Unknown
        #     return COLOUR_NOCOLOUR

    def __detect_code(self, code):

        # check for light blue
        min_HSV = np.array([12, 40, 90], dtype = "uint8")
        max_HSV = np.array([20, 255, 255], dtype = "uint8")
        image = self.image
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)

        skinHSV = cv2.bitwise_and(image, image, mask=skinRegionHSV)
        average = skinHSV.mean(axis=0).mean(axis=0)
        #print(f"Is comthing {average[2]}")
        # check if blue is there and have some value above threshold
        if average[2] < .4:
            self.sample.clear()
            return 0


        # check good/bad now
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)


        res = cv2.bitwise_and(image,image, mask= mask)

        average = res.mean(axis=0).mean(axis=0)
        if average[2] > 0 :
            print(f"Actual {average[2]}")

        if average[2] > 0.5 and average[2] < 7 : ## GOOD
            return 2
        else :
            return 1

        '''if code == COLOUR_DARKBLUE:
            if average[2] > 0.64: ## GOOD
                return 2
            
        if code == COLOUR_RED:
            ## Red
            min_HSV = np.array([0, 50, 50], dtype = "uint8")
            max_HSV = np.array([10, 255, 255], dtype = "uint8")
        elif code == COLOUR_LIGHTBLUE:
            ## Orange
            min_HSV = np.array([12, 40, 90], dtype = "uint8")
            max_HSV = np.array([20, 255, 255], dtype = "uint8")
            #min_HSV = np.array([100,50,50], dtype = "uint8")
            #max_HSV = np.array([105, 75, 255], dtype = "uint8")
        else:
            ## Dark Blue
            min_HSV = np.array([105,75,75], dtype = "uint8")
            max_HSV = np.array([180, 255, 255], dtype = "uint8")

        image = self.image
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)

        skinHSV = cv2.bitwise_and(image, image, mask=skinRegionHSV)
        average = skinHSV.mean(axis=0).mean(axis=0)
        #print(average[2])
        if average[2] > 0 :
            print(average[2])
        if code == COLOUR_DARKBLUE:
            if average[2] > 0.64: ## GOOD
                return 2
            if average[2] >0.2 and average[2] < 0.6: ## BAD
                return 1
        return 0
        try:
            skinHSV = cv2.medianBlur(skinHSV,5)
            gray = cv2.cvtColor(skinHSV, cv2.COLOR_BGR2GRAY)
            canny_edge = cv2.Canny(gray, 50, 240)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            canny_edge = cv2.dilate(canny_edge, kernel)
            circles = cv2.HoughCircles(canny_edge,cv2.HOUGH_GRADIENT, dp=1, minDist=35, param1=200, param2=15, minRadius=10, maxRadius=100)
            if circles is None:
                c = 0
            else:
                c = circles.size()
            print(f'Color {code} : HSV {average[2]} : Circles {c}')
        except:
            logger.error("Exception", exc_info =True)
        time.sleep(.4)
        return 0

        if circles is not None:
            ##
            return True
        return False'''

## Robotic Arm Sequences
    '''def __Bad_PCB_Posn(self):
        logger.debug("Picking Bad")
        swift = self.swift
        #swift.reset()
        #swift.set_speed_factor(0.2)
        swift.set_position(x=104, y=214, z=79)
        swift.set_position(relative = True, z = -77)
        swift.set_pump(on=True)
        swift.set_gripper(catch=True)
        swift.set_position(relative = True, z = 79)
        swift.set_position(x=286, y=70, z=79)
        swift.set_position(relative = True, z = -40)
        time.sleep(1)
        swift.set_pump(on=False)
        swift.set_gripper(catch=False)
        #swift.reset()
        swift.set_position(x=200, y=0, z=150)
        logger.debug("Placed Bad")
        
        swift.reset()
        swift.set_speed_factor(0.2)
        swift.set_position(x=104, y=214, z=79, speed=100000)
        swift.set_position(relative = True, z = -77)
        swift.set_pump(on=True)
        swift.set_gripper(catch=True)
        swift.set_position(relative = True, z = 100)
        swift.set_position(x=286, y=70, z=100, speed=100000)
        swift.set_position(relative = True, z = -40)
        time.sleep(1)
        swift.set_pump(on=False)
        swift.set_gripper(catch=False)
        swift.reset()
            
    def __Good_PCB_Posn(self):
        logger.debug("Picking Good")
        try:
            swift = self.swift
            #swift.reset()
            #swift.set_speed_factor(0.2)
            swift.set_position(x=104, y=214, z=79)
            swift.set_position(relative = True, z = -77)
            swift.set_pump(on=True)
            swift.set_gripper(catch=True)
            swift.set_position(relative = True, z = 79)
            swift.set_position(x=230, y=-79, z=79)
            swift.set_position(relative = True, z = -40)
            time.sleep(1)
            swift.set_pump(on=False)
            swift.set_gripper(catch=False)
            #swift.reset()
            swift.set_position(x=200, y=0, z=150)
            
            swift.reset()
            swift.set_speed_factor(0.2)
            swift.set_position(x=104, y=214, z=79, speed=100000)
            swift.set_position(relative = True, z = -77)
            swift.set_pump(on=True)
            swift.set_gripper(catch=True)
            swift.set_position(relative = True, z = 100)
            swift.set_position(x=175, y=-227, z=100, speed=100000)
            swift.set_position(relative = True, z = -40)
            time.sleep(1)
            swift.set_pump(on=False)
            swift.set_gripper(catch=False)
            swift.set_position(relative = True, z = 40)
            swift.reset()

        except:
            logger.error('[!] Swift...', exc_info = True)
        logger.debug("Placed Good")

    def __ManualCheck_PCB_Posn(self):
        logger.debug("Picking Manual Check")
        swift = self.swift
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
        logger.debug("Placed Manual Check")'''

