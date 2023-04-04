import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import scipy
from scipy.signal import correlate
import glob
from datetime import datetime
from itertools import chain
import imageio
from tqdm import tqdm
from PIL import Image
from MinervaManager import MinervaManager as MM
import time


def normalize_by_channel(image=None,normrows=None):
	'''Normalize by channel'''
	ch0mean = np.mean(image[normrows, :32])
	for ch in range(8):
		image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] / np.mean(image[normrows, ch*32:(ch+1)*32]) * ch0mean
	image = np.abs(image)
	return image

def remove_outliers(image=None,Nstd=5):
	'''Remove outlies'''
	med=np.median(np.ravel(image))
	std=np.std(np.ravel(image))
	image[np.abs(image-med)>(Nstd*std)] = med
	return image

def plot_single(data=None,data_name=None,tx=None,t0=None,colormap='Blues',verbose=True,vrange=[-4,1],normrows=None):
	fig = plt.figure(figsize=(12,6))
	grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2)
	ax_main = fig.add_subplot(grid[:, :])
	# im1 = ax_main.imshow(data, vmin=vrange[0], vmax=vrange[1], cmap=colormap)
	im1 = ax_main.imshow(np.flip(np.transpose(data),axis=1), #-np.median(image_1)), # [50:100,:40]),
									vmin=np.mean(data[normrows,:])+vrange[0]*np.std(data[normrows,:]), 
									vmax=np.mean(data[normrows,:])+vrange[1]*np.std(data[normrows,:]), 
									cmap=colormap)
	fig.colorbar(im1,ax=ax_main)
	ax_main.set_title(str(data_name)+ ' time elapsed ' + str(tx-t0))
	fig.canvas.draw()	   # draw the canvas, cache the renderer
	im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
	im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	if verbose:
		plt.show()
	plt.close(fig)
	return im

def save_animation(manager,savename,data_times=None,data_names=None,vrange=[-4,1],normrows=[200,300],colormap='Blues',verbose=True,fps=10):
	normrows=range(normrows[0],normrows[1])
	t0 = data_times[0]
	frames=[]
	for i in tqdm(range(len(data)), desc='Plotting'):
		im=plot_single(data=data[i],data_name=data_names[i],tx=data_times[i],t0=data_times[0],vrange=vrange,normrows=normrows,colormap='Blues',verbose=verbose)
		frames.append(im)
	print(' -- Saving plots as .gif animation -- ')
	plotdir = os.path.join(manager.logdir,'plots')
	if(not os.path.exists(plotdir)):
		os.mkdir(plotdir)
	filename=os.path.join(plotdir,savename.replace('.h5','.gif'))
	imageio.mimsave(filename,frames, fps=fps)
	print(' --  Animation saved as {}  -- '.format(filename))
	return 1

def imp_plot(manager=None,normrows=[0,511],vrange=[-4,1],mycolormap='Blues',Nstd=5,verbose=True,fps=10,deltas=False):
	'''Plots impedance timelapse from Minerva Logfiles.'''
	ph1,ph2,times,names = manager.get_data_stack()
	time.sleep(0.2)
	for lognum,logname in enumerate(manager.logfiles):
		fullname = os.path.join(manager.logdir,logname) 
		list_all = manager.get_list(fullname,sortby='time')
		list_impedance = manager.get_list(fullname,filterstring='impedance')
		if verbose:
			print('\n\nall\n\n',list_all)
			print('\n\nimpedance\n\n',list_impedance)
		t0 = manager.get_time(fullname,list_impedance[0])
		image_1_ref = None
		image_2_ref = None
		myframes=[]
		for i in tqdm(range(0,len(list_impedance),1),
			desc ='...Generating all impedance images from logfile {}'.format(os.path.basename(logname))):
			image_2d_ph1,image_2d_ph2 = manager.get_data(fullname,list_impedance[i])
			# ~~~~~~~~~~~~~~~~~~
			image_2d_ph1 = normalize_by_channel(image=image_2d_ph1,normrows=normrows)
			image_2d_ph2 = normalize_by_channel(image=image_2d_ph2,normrows=normrows)	
			# ~~~~~~~~~~~~~~~~~~
			image_2d_ph1 = remove_outliers(image=image_2d_ph1,Nstd=Nstd)
			image_2d_ph2 = remove_outliers(image=image_2d_ph2,Nstd=Nstd)	
			# ~~~~~~~~~~~~~~~~~~
			# re-normalize again
			image_2d_ph1 = normalize_by_channel(image=image_2d_ph1,normrows=normrows)
			image_2d_ph2 = normalize_by_channel(image=image_2d_ph2,normrows=normrows)	
			image_1 = image_2d_ph1
			image_2 = image_2d_ph2
			#Set Reference images to subtract for contrast
			if image_1_ref is None:
				image_1_ref = image_1
				continue
			if image_2_ref is None:
				image_2_ref = image_2
				continue
			# Subtract first phase image from dataset if there are artifacts in the image
			if deltas:
				if i!=0:
					image_1-=image_1_ref
					image_2-=image_2_ref
			final_image = image_1
			tx = manager.get_time(fullname,list_impedance[i])
			fig = plt.figure(figsize=(12,6))
			grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2)
			ax_main = fig.add_subplot(grid[:, :])
			im1 = ax_main.imshow(np.flip(np.transpose(final_image),axis=1), #-np.median(image_1)), # [50:100,:40]),
								vmin=np.mean(final_image[normrows,:])+vrange[0]*np.std(final_image[normrows,:]), 
								vmax=np.mean(final_image[normrows,:])+vrange[1]*np.std(final_image[normrows,:]), 
								cmap=mycolormap)
			fig.colorbar(im1,ax=ax_main)
			ax_main.set_title(str(lognum) + '   ' + str(i) + '   ' + list_all[i] + ' time elapsed ' + str(tx-t0))
			if verbose:
				plt.show()
			# add to frames for animation
			fig.canvas.draw()	   # draw the canvas, cache the renderer
			im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			myframes.append(im)
			plt.close(fig)
		#Save all frames
		print(' -- Saving plots as .gif animation -- ')
		plotdir = os.path.join(manager.logdir,'plots')
		if(not os.path.exists(plotdir)):
			os.mkdir(plotdir)
		savename=os.path.join(plotdir,os.path.basename(logname).replace('.h5','.gif'))
		imageio.mimsave(savename,myframes, fps=fps)
		print(' --  Animation saved as {}  -- '.format(savename))
	return 1 

def misc():
	pass