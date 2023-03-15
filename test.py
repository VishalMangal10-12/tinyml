#!/usr/bin/env python
import serial
import time
import argparse
import threading

writeSerial = False

def main():
    global writeSerial
    while True :
        print("Stopping belt")
        writeSerial = True
        time.sleep(15)
        print("Starting belt")
        writeSerial = False
        time.sleep(30)

def writeSerial():
    ser = serial.Serial()  # open serial port
    ser.baudrate = 115200
    ser.port = 'COM4'
    ser.open()

    if ser.is_open:
        print("Port open {}".format(ser.name))
        #ser.write(b'B')     # write a string
        while True:
            if writeSerial :
                ser.write(b'B')
        ser.close()
    else:
        print("Could not open serial")       # check which port was really used

if __name__ == "__main__":
    pipeline = threading.Thread(target=writeSerial, args=())
    pipeline.daemon = True
    pipeline.start()
    main()
    pipeline.join()
    

