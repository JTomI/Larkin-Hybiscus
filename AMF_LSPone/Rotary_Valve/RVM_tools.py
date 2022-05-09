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
		# Open serial connection -> check COM port on your device
		self.rvm = Serial(port, 9600, timeout=1000)
		print('RVM connected on',self.rvm.name)
		self.rvm.write(b"/1?6R\r")
		msg = self.rvm.readline()
		print("RVM initialised. Current valve = "+str(msg))
		self.rvm.close()
		# sys.exit(0)

	def query(self, cmd='?6', rbytes=100):
		self.rvm.open()
		cmd_str = "/1{}R\r".format(cmd)
		self.rvm.write(cmd_str.encode('utf-8'))
		msg = self.rvm.readline()
		self.rvm.close()
		return msg

	def get_ID(self):
		self.rvm.open()
		self.rvm.write(b"/1?9000R\r")
		id_ = self.rvm.readline()
		print("RVM unique ID = ",id_)
		self.rvm.close()
		return id_

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
		valv_num=self.rvm.write(b"/1?26R\r")
		print("Homing complete. Current valve = "+str(valv_num))
		self.rvm.close()
		return valv_num

	def move(self,valve=None):
		# Moves to new valve position (1,2,3,4,5,6) in shortest path.
		# /1 	- command start
		# b3 	- go to valve port 3, shortest path, unless already at valve 3
		# R 	- command end
		if valve == None:
			print("No valve (1,2,3,4,5,6) selected")
		else:
			self.rvm.open()
			cmd_str = "/1b{}R\r".format(int(valve))
			rvm.write(cmd_str.encode('utf-8'))
			time.sleep(10)
			valv_num=self.rvm.write(b"/1?26R\r")
			print("Move complete. Current valve = "+str(valv_num))
			self.rvm.close()
			return valv_num

	def exit(self):
		#Dramatic program end
		sys.exit(0)


if __name__ == '__main__':
	# Example connection and use. 
	rvm1 = RVM()
	# rvm1.get_ID()
	print(rvm1.query())
	# rvm1.reset()
	# rvm1.move(valve=1)

	#Make sure to close connection!
	sys.exit(0)