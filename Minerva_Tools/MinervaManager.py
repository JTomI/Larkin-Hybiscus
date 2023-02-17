import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import glob
from datetime import datetime
from itertools import chain
import imageio


class MinervaManager(object):
	'''Manager class with methods for importing and basic handling of data/metadata in Minerva generated .h5 logfiles.'''

	def __init__(self,dtype='imp',logdir=None):
		'''Manager is initialized to handle all logfiles sharing one minerva datatype (dtype), that exist in directory at absolute path (logdir). 
		Compatible dtype values are imp (Impedance), ect (Electrical Capacitance Tomography), ph (pH)'''
		self.filters = {'imp':'impedance','ph':'pH','ect':'ECT'}
		self.dtype = dtype
		self.logdir = logdir
		self.logfiles = glob.glob(logdir+'*{}*.h5'.format(dtype))
		self.filterstring = self.filters[dtype]
		print('Logfile Directory: ',logdir)
		print(' -- ({}) Logfiles of *{}* datatype -- '.format(len(self.logfiles),self.filters[dtype]))
		print('')
		for filename in self.logfiles:
			imp_list = self.get_list(filename)
			print('Filename: ',os.path.basename(filename))
			print('# images: ',len(imp_list))
			print('')

	def get_data(self,filename=None, exp_name=None, dataname='image'):
		'''Searches for a dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the image data (dataname) of the dataset as a 2D numpy array.'''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		image = grp_data[dataname][:]
		return image

	def all_attr(self,filename=None, exp_name=None):
		'''Searches for a dataset (exp_name) stored in a Minerva Logfile (filename). 
		Returns h5py object with all of the dataset's metadata, as well as a list of the attribute names. '''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		return grp_data.attrs, list(grp_data.attrs.keys())

	def get_attr(self,filename=None, exp_name=None, attrname=None):
		'''Searches a Dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the value of the specified metadata attribute (attrname).'''
		all_attr,attr_list = self.all_attr(filename=filename,exp_name=exp_name)
		return all_attr[attrname]

	def get_time(self,filename=None, exp_name=None):
		'''Searches a Dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the timestamp of the dataset.'''
		return datetime.strptime(self.get_attr(filename, exp_name, 'timestamp'), "%Y%m%d_%H%M%S")

	def get_list(self,filename=None, filterstring=None, sortby='time'):
		if filterstring==None:
			filterstring = self.filterstring
		hf = h5py.File(filename, 'r')
		base_items = list(hf.items())
		grp_list = []
		for i in range(len(base_items)):
			grp = base_items[i]
			grp_list.append(grp[0])
		if filterstring is not None:
			grp_list = [x for x in grp_list if filterstring in x]
		if sortby is 'time':
			grp_list = sorted(grp_list,key=lambda x: self.get_time(filename,x))
		return grp_list


if __name__ == '__main__':
	pass