#Run following command first in anaconda env
#conda develop /{path}/{to}/{MinervaScripts}
import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import glob
import ray
from datetime import datetime
from itertools import chain
import imageio
from tqdm import tqdm

#Ray doesnt like to pickle functions if they arent static, i.e. outside class

class MinervaManager(object):
	'''Manager class with methods for importing and basic handling of data/metadata in Minerva generated .h5 logfiles.'''
	def __init__(self,logdir=None):
		'''Manager is initialized and finds all .h5 files that exist in directory at absolute path (logdir), and sort into dict by minerva datatype (dtype), . 
		Compatible dtype values are imp (Impedance), ect (Electrical Capacitance Tomography), ph (pH)'''
		# Filters sets up the data types from .h5 that can be handled
		self.filters = {'imp':'impedance','ph':'pH','ect':'ECT'}
		self.logdir = logdir
		self.logfiles={}
		self.logfile_explists={}
		print('Logfile Directory: ',logdir)
		# Find all files in logdir of each datatype that can be handled by manager
		for dtype in self.filters.keys():
			mode_logfiles = glob.glob(logdir+'*{}*.h5'.format(dtype))
			print(' -- ({}) Logfiles of *{}* datatype -- '.format(len(mode_logfiles),self.filters[dtype]))
			self.logfiles[dtype]=mode_logfiles
		print('')
		for dtype in self.filters.keys():
			mode_explists=[]
			for filename in self.logfiles[dtype]:
				explist = self.get_list(filename, filterstring=self.filters[dtype])
				print('Filename: ', os.path.basename(filename))
				print('# images: ', len(explist))
				#Save all experiment names in a separate list for each logfile
				mode_explists.extend(explist)
			#Enter experiment lists into dictionary under their dtype
			self.logfile_explists[dtype]=mode_explists

	# Basic functions, for manually pulling up data directly from h5. 
	def get_raw_data(self, exp_name=None,filename=None, dataname='image'):
		'''Manual search for a dataset (exp_name) stored in a Minerva Logfile (filename).
		Returns the specified image data (dataname) of that dataset as a 2D numpy array.'''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		image = grp_data[dataname][:]
		return image

	def get_data(self,filename=None, exp_name=None):
		'''Searches for a Dataset (exp_name) stored in a Minerva .h5 Logfile (filename).
		Returns both phases of the the image dataset as 2D numpy arrays.
		Image data is preprocessed to apply overall gain to image.'''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		image_2d_ph1 = grp_data['image_2d_ph1'][:]
		image_2d_ph2 = grp_data['image_2d_ph2'][:]
		return image_2d_ph1 , image_2d_ph2

	def all_attr(self,filename=None, exp_name=None):
		'''Searches for a Dataset (exp_name) stored in a Minerva .h5 Logfile (filename). 
		Returns h5py object with all metadata attributes, as well as a list of the attribute names. '''
		hf = h5py.File(filename, 'r')
		grp_data = hf.get(exp_name)
		return grp_data.attrs, list(grp_data.attrs.keys())

	def get_attr(self,filename=None, exp_name=None, attrname=None):
		'''Searches for a Dataset (exp_name) stored in a Minerva .h5 Logfile (filename).
		Returns the value of the specified metadata attribute (attrname).'''
		all_attr,attr_list = self.all_attr(filename=filename,exp_name=exp_name)
		return all_attr[attrname]

	def get_time(self,filename=None, exp_name=None):
		'''Searches for a Dataset (exp_name) stored in a Minerva .h5 Logfile (filename).
		Returns the timestamp of that dataset.'''
		return datetime.strptime(self.get_attr(filename, exp_name, 'timestamp'), "%Y%m%d_%H%M%S")

	def get_list(self, filename=None, filterstring=None, sortby='time'):
		'''Finds all the names of experiments in a .h5 file matching the specified datatype,
		 returns them in a time ordered list.

		 '''
		with h5py.File(filename, 'r') as hf:
			base_items = list(hf.items())
		grp_list = [grp[0] for grp in base_items]
		if filterstring is not None:
			grp_list = [x for x in grp_list if filterstring in x]
		if sortby == 'time':
			grp_list = sorted(grp_list,key=lambda x: self.get_time(filename,x))
		return grp_list

	# Functions for higher level data handling. 
	def process_file(self, file_path, exp_name):
		'''Function meant to enable the import of each image via list comprehension in get_data_stack'''
		image_2d_ph1, image_2d_ph2 = self.get_data(file_path, exp_name)
		return (image_2d_ph1 + image_2d_ph2)/2

	def get_data_stack(self, dtype='imp', nrow=512, ncol=256):
		'''Detects all .h5 files in working directory and compiles a time ordered image
		  sequence of all data in that directory of the specified data type. 
		  Can handle compilation of impedance/ph timelapses, or ECT stacks. 
		  Multiphase readouts are averaged together (image_2d_ph1 + image_2d_ph2)/2 before return

		Keyword arguments:
		dtype (str) -- The function grab all data corresponding to 'dtype' from the working directory,  returns it as t-ordered. 
				Available dtype options are in self.filters.keys(), i.e. 'imp', 'ect', 'ph'.
		nrow/ncol (int) -- the anticipated dimensions of the image data. Default is Minerva Dimensions.

		Return arguments:
		frames (3D array with float elements) -- t-ordered image sequence of shape (nimages,nrow,ncol). 
		timestamps (list of datetime objects) -- t-ordered list of the timestamps associated with frames array, shape (nimages,)
		file_paths_exp_names (list of tupled strings) -- t-ordered list of tupled (filepaths, frame names) associated with frames array, shape (nimages,2)'''
		file_paths_exp_names = [(os.path.join(self.logdir, logname), exp) 
					for logname in self.logfiles[dtype] for exp in self.get_list(os.path.join(self.logdir, logname), 
						filterstring=self.filters[dtype], sortby='time')]
		num_images = len(file_paths_exp_names)
		print(f' -- (fn:get_data_stack) Importing {num_images} {self.filters[dtype]} images from ({len(self.logfiles[dtype])}) logfiles -- ')
		timestamps = [self.get_time(file_path, exp_name) for file_path, exp_name in file_paths_exp_names]
		frames = np.zeros((num_images, nrow, ncol), dtype='float')
		frames[:] = np.array([self.process_file(file_path, exp_name) for file_path, exp_name in file_paths_exp_names])
		# Get the size in Bytes of the full dataset
		total_size = frames.nbytes + sys.getsizeof(timestamps)+sys.getsizeof(file_paths_exp_names)
		print('Total import size: ', total_size, 'Bytes')
		return frames, timestamps, file_paths_exp_names

if __name__ == '__main__':
	# Example usage, fetching impedance data from working directory impdir
	impdir = r"C:\Users\jtincan\Desktop\F0386_Analysis\F0386_minerva\impedance/"
	iM = MinervaManager(logdir=impdir);
	# Logfile Directory:  C:\Users\jtincan\Desktop\F0386_Analysis\F0386_minerva\impedance/
	# -- (2) Logfiles of *impedance* datatype -- 

	# Filename:  F0386_imp_p1.h5
	# # images:  18
	# Filename:  F0386_imp_p2.h5
	# # images:  1800

	#Get first .h5 log file
	logname = iM.logfiles[0]
	#Get full filepath of that log file
	filename = os.path.join(iM.logdir,logname)
	#Get list of experimental datasets from this first logfile 
	explist = iM.logfile_explists[0]
	#Fetch the phase pair of impedance images from first timepoint of the first log file
	image_2d_ph1,image_2d_ph2 = iM.get_data(filename,explist[0])
	#Fetch a stack of all impedance type images in working directory from first timepoint of the first log file.
	#Phase pairs are already combined (averaged) into single stack. Also outputs timestamps and tupled filepaths/expnames
	imp_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'imp')
	#Fetch a stack of all ECT type images in working directory from first timepoint of the first log file.
	ect_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'ect')
	#Fetch a stack of all ph type images in working directory from first timepoint of the first log file.
	ect_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'ph')
