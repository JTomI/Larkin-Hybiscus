#conda develop /path/to/MinervaScripts
import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import glob
from datetime import datetime
from itertools import chain
import imageio
from tqdm import tqdm

class MinervaManager(object):
	'''Manager class mLwith methods for importing and basic handling of data/metadata in Minerva generated .h5 logfiles.'''
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
		self.logfile_explists = []
		for filename in self.logfiles:
			explist = self.get_list(filename)
			print('Filename: ',os.path.basename(filename))
			print('# images: ',len(explist))
			#Save all experiment names in a separate list for each logfile
			self.logfile_explists.append(explist)

	def get_raw_data(self,filename=None, exp_name=None, dataname='image'):
		'''Searches for a dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the image data (dataname) of the dataset as a 2D numpy array.'''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		image = grp_data[dataname][:]
		return image

	def get_data(self,filename=None, exp_name=None):
		'''Searches for a dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the image data (dataname) of the dataset as a 2D numpy array.
		Data preprocessed to apply overall gain to image'''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		image_2d_ph1 = grp_data['image_2d_ph1'][:]
		image_2d_ph2 = grp_data['image_2d_ph2'][:]
		# V_SW = self.get_attr(filename,exp_name,'V_SW')
		# V_CM = self.get_attr(filename,exp_name,'V_CM')
		# f_sw = self.get_attr(filename,exp_name,'f_sw')
		# T_int = self.get_attr(filename,exp_name,'T_int')
		# C_int = self.get_attr(filename,exp_name,'C_int')
		# if (not self.dtype == 'ph'):
		# 	gain_swcap = np.abs(V_SW-V_CM)*1e-3*f_sw  # Iout/Cin
		# 	gain_integrator = T_int/C_int  # Vout/Iin
		# 	gain_overall = gain_swcap*gain_integrator
		# 	image_2d_ph1 = image_2d_ph1 / gain_overall
		# 	image_2d_ph2 = image_2d_ph2 / gain_overall
		return image_2d_ph1 , image_2d_ph2

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

	def get_data_stack(self,imrange=None):
		'''Returns 4 time ordered lists of ph, ect, or impedance data,
		 with one list for each collected phase, one list of timestamps and one list of frame names.
		Function iterates over each logfile in the directory and compiles 
		all image data from all logfiles with data type self.dtype.'''
		frames_ph1=[]
		frames_ph2=[]
		timestamps=[]
		exp_names=[]
		for lognum,logname in enumerate(self.logfiles):
			fullname = os.path.join(self.logdir,logname)
			list_all = self.get_list(fullname,filterstring=self.filterstring,sortby='time')
			if imrange==None:
				imrange=[0,len(list_all)]
			for i in tqdm(range(imrange[0],imrange[1]),
			 desc=' --  Importing {} data from {}  -- '.format(self.filterstring,os.path.basename(logname))):
				image_2d_ph1,image_2d_ph2 = self.get_data(fullname,list_all[i])
				frames_ph1.append(image_2d_ph1)
				frames_ph2.append(image_2d_ph2)
				timestamps.append(self.get_time(fullname,list_all[i]))
				exp_names.append(list_all[i])
		print('Completed import of {} {} images from ({}) logfiles'.format(len(frames_ph1),self.filterstring,len(self.logfiles)))
		print('Total import size: ',sys.getsizeof(frames_ph1)+sys.getsizeof(frames_ph2)+sys.getsizeof(timestamps)+sys.getsizeof(exp_names),'Bytes')
		return frames_ph1, frames_ph2, timestamps, exp_names


if __name__ == '__main__':
	# Example usage, fetching impedance data
	impdir = r"C:\Users\jtincan\Desktop\F0386_Analysis\F0386_minerva\impedance/"
	iM = MinervaManager(dtype='imp',logdir=impdir);
	# Logfile Directory:  C:\Users\jtincan\Desktop\F0386_Analysis\F0386_minerva\impedance/
	# -- (2) Logfiles of *impedance* datatype -- 

	# Filename:  F0386_imp_p1.h5
	# # images:  18
	# Filename:  F0386_imp_p2.h5
	# # images:  1800

	#Get first .h5 log file
	logname = iM.logfiles[0]
	#Get full filepath
	filename = os.path.join(iM.logdir,logname)
	#Get capture/'experiment' list of first logfile 
	datalist = iM.logfile_explists[0]
	#Fetch pair of impedance images from first timepoint of the first log file
	image_2d_ph1,image_2d_ph2 = iM.get_data(filename,datalist[0])
