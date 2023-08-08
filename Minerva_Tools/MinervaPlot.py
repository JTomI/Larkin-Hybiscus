import h5py
from matplotlib import font_manager
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
from PIL import Image, ImageOps
from MinervaManager import MinervaManager as MM
import time

def rm_banding(image=None,normrows=[0,511]):
	'''Removes banding artifact from channel readout, while preserving mean pixel value in image. 
	Should be used after get_data function before other processing'''
	# Record the full frame average of the image before removal
	fullframeaverage=np.mean(image[normrows[0]:normrows[1],:])
	# Get the profile of the banding
	rowaverage=np.mean(image[normrows[0]:normrows[1],:],axis=0)
	# Make mask of banding and subract it off of image
	band_mask = np.ones_like(image[normrows[0]:normrows[1],:])
	axislen = len(band_mask[:,0])
	for i in range(axislen):
		band_mask[i,:]=rowaverage
	image[normrows[0]:normrows[1],:]-=band_mask
	# Add back in original frame average to preserve mean
	image[normrows[0]:normrows[1],:]+=fullframeaverage
	return image

def normalize_by_channel(image=None,normrows=None):
	'''Normalize by channel'''
	ch0mean = np.mean(image[normrows, :32])
	for ch in range(8):
		image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] / np.mean(image[normrows, ch*32:(ch+1)*32]) * ch0mean
	image = np.abs(image)
	return image

def remove_outliers(image=None,Nstd=5):
	'''Remove outlies'''
	med=np.mean(np.ravel(image))
	std=np.std(np.ravel(image))
	image[np.abs(image-med)>(Nstd*std)] = med
	return image

def vrange_crop(image=None, vrange=[-4,1]):
	med=np.mean(np.ravel(image))
	std=np.std(np.ravel(image))
	image[(image-med)<=(vrange[0]*std)] = vrange[0]*std
	image[(image-med)>=(vrange[1]*std)] = vrange[1]*std
	return image

def plot_single(data=None,data_name='',tx=0,t0=0,colormap='Blues',verbose=True,vmin=None,vmax=None,vrange=[-4,1],normrows=None):
	'''Image should display with orientation such that minerva PCB is to the left'''
	fig = plt.figure(figsize=(12,6))
	grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2)
	ax_main = fig.add_subplot(grid[:, :])
	if (vmin==None and vmax==None):
		vmin=np.mean(data[normrows,:])+vrange[0]*np.std(data[normrows,:])
		vmax=np.mean(data[normrows,:])+vrange[1]*np.std(data[normrows,:])
	im1 = ax_main.imshow(data, vmin=vmin, vmax=vmax, cmap=colormap)
	fig.colorbar(im1,ax=ax_main)
	# ax_main.set_title(str(data_name)+ ' time elapsed ' + str(tx-t0))
	fig.canvas.draw()	   # draw the canvas, cache the renderer
	im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
	im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	if verbose:
		plt.show()
	plt.close(fig)
	return im

def image_timelapse(manager,savename,data_times=None,data_names=None,vrange=[-4,1],normrows=[200,300],colormap='Blues',verbose=True,fps=10):
	'''Saves impedance data as a pure image timelapse, no labels/graphs etc.'''
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

def graph_timelapse(manager=None,savename=None,images=None, timestamps=None,imrange=None,vmin=None,vmax=None,mycolormap='Blues',fps=10,verbose=False):
    '''Plots impedance data in graphs then saves as a timelapse in .gif video format.'''
    t0 = timestamps[0]
    if imrange==None:
        imprange=range(len(images))
    else:
        imprange=range(len(images[imrange[0]:imrange[1]]))
    myframes=[]
    for i in tqdm(imprange, desc =' -- Generating impedance timelapse -- '):
        tx=times[i]
        fig = plt.figure(figsize=(12,6))
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2)
        ax_main = fig.add_subplot(grid[:, :])
        im1 = ax_main.imshow(np.flip(images[i]),vmin=vmin,vmax=vmax, cmap=mycolormap)
        cb=fig.colorbar(im1,ax=ax_main,label='Capacitance [Farads]')
        axcb = cb.ax
        text = axcb.yaxis.label
        font = font_manager.FontProperties(family='times new roman', style='italic', size=20)
        text.set_font_properties(font)
        ax_main.set_title('Time Elapsed: ' + str(tx-t0))
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
    path =os.path.join(plotdir,savename+'.gif')
    imageio.mimsave(path,myframes, fps=fps)
    print(' --  Animation saved as {}  -- '.format(savename))
    return 1  


def get_hist(data=None,bins=256,figsize=(12,8),verbose=False):
	counts,bins=np.histogram(data, bins=bins);
	places = np.where(counts>1);
	vmin_i,vmax_i= np.min(places),np.max(places);
	(vmin,vmax) = (bins[:-1][vmin_i],bins[:-1][vmax_i])
	if verbose:
		plt.figure(figsize=figsize);
		plt.title('Extracting vmin/vmax to set contrast');
		plt.hist(bins[:-1], bins, weights=counts,label='Counts');
		plt.hist( [bins[:-1][vmin_i],bins[:-1][vmax_i]], bins, weights=[np.max(counts),np.max(counts)],label='Histogram Edges');
		plt.legend();
		print('(vmin,vmax)=',(vmin,vmax));
	return vmin,vmax

def combine_ph(images_ph1,images_ph2):
	'''Standard function for combining impedance images of different phase.'''
	final_images = (np.array(images_ph1)+np.array(images_ph2))/2
    for i in tqdm(range(len(final_images)),desc= ' -- Combining impedance phases -- '):
        final_images[i] = remove_outliers(final_images[i])
        final_images[i] = rm_banding(final_images[i])
    return final_images