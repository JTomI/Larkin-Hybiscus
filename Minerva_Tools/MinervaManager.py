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
import tifffile as tif
from tqdm import tqdm

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
		nrow/ncol (int) -- The anticipated dimensions of each image. Default is Minerva Dimensions.

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

	def array_to_tiff(self, images=None, savepath=None,normto=255, astype=np.float64):
		'''Function which saves an image sequence in a 3D array as a .tiff stack. 
		Primarily intended for saving impedance arrays as .tiff, but should work for numpy arrays.

		Keyword arguments:
		images (3D array) -- t-ordered numpy array representing image sequence to be saved. Shape = (t, nrows,ncols)
		savepath (str) -- Absolute path to save the .tiff file to. Default "None" trys to save to MinervaManager's working directory.
		astype (str) -- Image Sequence is saved with pixel values cast to type astype. Default float64 preserves image bitdepth.
		normto (int) -- Scaling factor to renormalize pixel values in the image sequence, used to make images more FiJi friendly.
							Default renormalized to pixel value range 0->255.
		
		Return arguments:
		filepath (str) -- The path to the saved .tiff
		'''
		# Indicate to tif.imsave to indicate if filesize to be saved as .tiff exceeds 2GB
		if int(images.nbytes) >= 2e+9:
			bigtiff=True
		else:
			bigtiff=False
		# Default trye to save save .tiff in new folder in working directory
		if savepath == None:
			plotdir = os.path.join(self.logdir,'tiff_stacks')
			if(not os.path.exists(plotdir)):
				os.mkdir(plotdir)
			print(plotdir)
			for filelist in list(self.logfiles.values()):
			    if filelist:
			        filename = filelist[0].replace('.h5','')
			        break
			print(filename)
			filename = os.path.basename(filename)[0:5]
			print(filename)
			savepath = os.path.join(plotdir,filename+'.tiff')
		im_max=np.max(images)
		tif.imwrite(savepath, ((normto/im_max)*images).astype(astype), bigtiff=bigtiff)
		print(f' -- (fn:array_to_tiff) {images.shape[0]} image sequence saved as {savepath}. Pixel values were renormalized to range [0,{normto}] -- ')
		return savepath

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
	#Fetch a stack of all of one type images of images in working directory from first timepoint of the first log file.
	imp_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'imp')
	ect_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'ect')
	ph_frames, timestamps, file_paths_exp_names = iM.get_data_stack(dtype = 'ph')
