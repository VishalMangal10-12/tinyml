#!/usr/bin/env python
import serial
import time
from flask import Response
from flask import Flask
from flask import render_template
import threading


# initialize a flask object
app = Flask(__name__)
ser = serial.Serial() # open serial port
ser.baudrate = 115200
ser.port = 'COM4'
ser.open()

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/start")
def good():
    global ser
    if ser.is_open:
        print("Port start open {}".format(ser.name))
        ser.write(b'M') # write a string
    else:
        print("Could not open serial") # check which port was really used
    return "Ok"

@app.route("/stop")
def bad():
    global ser
    if ser.is_open:
        print("Port stop open {}".format(ser.name))
        ser.write(b'S') # write a string
    else:
        print("Could not open serial") # check which port was really used
    return "Ok"

def writeSerial():
    global ser
    while True:
        if ser.is_open:
            data = ser.read()
            print(data)
        else:
            print("Could not open serial")       # check which port was really used

pipeline = threading.Thread(target=writeSerial, args=())
pipeline.daemon = True
pipeline.start()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    pipeline.join()
