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
from serial import Serial
import time

class RVM(object):
	def __init__(self,port='COM3'):
		self.port=port
		self.current_valve=0
		# Open serial connection -> check COM port on your device
		print("RVM initializing")
		self.rvm = Serial(port, 9600, timeout=10)
		print('RVM connected on',self.rvm.name)
		# Close the connection until it is needed again
		self.rvm.close()

	def cmd(self, msg=''):
		# Send arbitrary command to RVM. Consult AMF manual for the RVM commands.
		# /1 	- command start
		# {} 	- command body
		# R 	- command end
		# \r    - carriage return caracter
		self.rvm.open()
		cmd_str = "/1{}R\r".format(msg)
		self.rvm.write(cmd_str.encode('utf-8'))
		self.rvm.close()

	def reset(self):
		#Reset connection
		self.rvm = Serial(self.port, 9600, timeout=1000)
		print("Connection reset on port "+str(self.rvm.name))
		self.rvm.close()

	def rehome(self):
		#Perform rehoming of RVM
		self.rvm.open()
		self.rvm.write(b"/1ZR\r")
		time.sleep(5)
		print("Homing complete.")
		self.rvm.close()

	def move(self,valve=None,delay=1):
		# Moves to new valve position (1,2,3,4,5,6) in shortest path.
		# /1 	- command start
		# b{} 	- move to valve port {} by the shortest path, unless already at that valve.
		# R 	- command end
		# \r    - carriage return caracter
		if valve == None:
			print("No valve (1,2,3,4,5,6) selected")
		else:
			self.rvm.open()
			cmd_str = "/1b{}R\r".format(int(valve))
			self.rvm.write(cmd_str.encode('utf-8'))
			self.current_valve=valve
			time.sleep(delay)
			print("Move complete. Current valve = "+str(self.current_valve))
			self.rvm.close()


if __name__ == '__main__':
	# Example connection and use. 
	rvm1 = RVM()
	rvm1.rehome()
	print(rvm1.current_valve)
	rvm1.move(valve=1)
	print(rvm1.current_valve)
	rvm1.move(valve=6)
	print(rvm1.current_valve)
	rvm1.reset()
	#Make doubly that connection is closed
	sys.exit(0)