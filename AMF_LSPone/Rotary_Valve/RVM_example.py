"""
RVM_example.py
~~~~~~~~~~~~~~~~~
This program shows how to connect to to the RVM (OEM rotary valve) using python.
The different commands can be found in the user manual.

:copyright: (c) 2017, E. Collot
:license: Proprietary, see LICENSE for details.

"""
#include python libraries
import sys
import serial
import time

# Open serial connection -> check COM port on your device
rvm = serial.Serial('COM7', 9600, timeout=1000)
print('RVM connected on ',rvm.name)

# Initialise RVM
rvm.write(b"/1ZR\r")
time.sleep(5)
rvm.write(b"/1O3R\r")
time.sleep(10)
print("RVM ready")

#Here is the detail
# /1 	- command start
# O3 	- go to port 3, clockwise rotation
# R 	- command end

#Finishing the script 
sys.exit(0)

