import h5py
from matplotlib import font_manager
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from copy import deepcopy
import glob
from datetime import datetime
from itertools import chain
import imageio
import ray
from tqdm import tqdm
from PIL import Image, ImageOps
from MinervaManager import MinervaManager as MM
import time
from skimage.feature import canny
from skimage.filters import sobel, threshold_multiotsu
from skimage.segmentation import watershed
import scipy
from scipy.signal import argrelextrema, find_peaks, square, decimate, correlate
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy import fftpack
from pylab import cm

#------------------------------------ Image Processing Methods --------------------------------
def cleanup(images,Nstd=5):
	'''Standard function for removing debanding effect in impedance/ECT stacks, and removing outliers.'''
	print(f' -- (fn:cleanup) Removing banding and outliers > {Nstd} sigma from mean -- ')
	images = remove_outliers(images,Nstd=Nstd)
	images = rm_banding(images)
	print(f'Cleaned up {images.shape[0]} images')
	return images

def rm_banding(images=None):
	'''Removes channel readout banding artifact from impedance or ECT stack. Assumes input shape (n,rows,cols).
	Preserves global mean of each nth image.'''
	frame_averages = np.mean(images, axis=(1, 2), keepdims=True)# Get the average of each image, broadcast to shape (n,512,256)
	col_averages = np.mean(images, axis=1, keepdims=True)
	# Subtract a broadcasted column wise average to remove banding. Then add back in the frame average to preserve each frame's global mean
	return images + frame_averages - col_averages

def remove_outliers(images=None, Nstd=5):
	'''Suppress outliers above N standard deviations from the mean of each image by setting their values to the mean.'''
	# Process each image in stack individually
	for i in range(images.shape[0]):
		image = images[i,:,:] 
		original_shape = image.shape #Save the original shape
		mean = np.mean(np.ravel(image))#get mean&std per image
		std = np.std(np.ravel(image))
		image[np.abs(image - mean) > Nstd * std] = mean # Replace outliers per image
		images[i] = image.reshape(original_shape)# Retrun to original shape
	return images

def normalize_by_channel(image=None,normrows=None):
	'''Normalize by channel. Deprecated, use rm_banding or cleanup methods. '''
	ch0mean = np.mean(image[normrows, :32])
	for ch in range(8):
		image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] / np.mean(image[normrows, ch*32:(ch+1)*32]) * ch0mean
	image = np.abs(image)
	return image

def imp_downsample(images=None,n=1):
	'''Downsample an impedance dataset of shape (nimages,rows,cols) by averaging each n elements of array along first dimension.'''
	num_chunks = images.shape[0]//n
	reshaped_images = images[:num_chunks * n].reshape((num_chunks, n) + images.shape[1:])
	# Take the mean along the second axis (axis=1) to get the ordered mean
	return reshaped_images.mean(axis=1)

def meta_downsample(meta_data=None,n=1):
	'''Downsample a list or 1Darray by just slicing to keep every nth element along first axis'''
	if type(meta_data)==list:
		meta_data = np.array(meta_data)
	return meta_data[::n]

def detect_otsu(image=None,nclass=3):
	'''Masks an image according to an n-class multi-otsu classification.

	Keyword arguments:
	image (2D array) -- The image data to be classified
	nclass (int) -- The number of classes to assign image data to

	Return arguments:
	otsumask (int 2D array) -- Array matching shape of image, containing the classification results for each pixel.
							Similar pixels will share an integer label with values ranging from 0 to nclass-1.
	thresholds (list) -- List containing a number of thresholds (N=nclass-1) used to classify the pixels in image.
	'''
	thresholds = threshold_multiotsu(image,classes=nclass)
	otsumask = np.digitize(image, bins=thresholds)
	return otsumask, thresholds

def largest_feature(binary_image=None):
	''' Find the largest contiguous feature in a binary image.

	Keyword arguments:
	binary_image (bool 2D array) -- A binary image with some number of clusters

	Return arguments:
	max_feature (bool 2D array) -- A new binary image containing only the largest cluster
	labeled_image (int 2D array) -- A labeled version of the input image, where the pixels of each cluster 
										have been given an integer value indicating their grouping. 
	'''
	labeled_image, num_features = ndi.label(binary_image) # Label connected components in the binary image  
	labeled_feature_sizes = ndi.sum(binary_image, labels=labeled_image, index = range(1, num_features + 1))
	max_feature = np.zeros_like(binary_image)
	max_feature[labeled_image==largest_feature_label]=1 # Extract the largest connected component
	return max_feature , labeled_image

def fft_mask(images=None, linewidth=10, recoverywidth=100, gridshape=(511,256)):
	'''Generates a mask in frequency domain that suppressed periodic features of the CMOS in optical images. 
	(col, row) features of the Minerva CMOS show up at (512,256) pixel intervals in optical data.

	Keyword arguments:
	images (2D or 3D array) -- Image sequence of shape (nrow, ncol) or (nimages,nrow,ncol) 
	linewidth (int) -- Along each harmonic of CMOS pattern, the FFT value is set to zero with a mask. High linewidth
						value improves suppression of less coherent harmonics but loses signal, and vice-versa.
	recoverywidth (int) -- Low frequency in FFT usally correlates with image signal you want. The low f region spanning 
							(dfx,dfy) = (0->2*recoverywidth, 0->recoverywidth) is not suppressed
	gridshape (tuple int) -- The 2D shape of the underlying CMOS array appearing in the optical image

	Return arguments:
	mask (bool 2D array) -- A 2D binary mask, of same x,y shape as input 'images'. Corresponds to
						 FFT regions s.t. grid features are suppressed w.r.t. useful image signal. 
	'''
	ncols,nrows = gridshape
	if len(images.shape)==3: # Handle single images as well as stacks
		mask=np.ones_like(images[0,:,:])
	else:
		mask = np.ones_like(images)
	# mask out the harmonic frequencies associated with 512/256 pixel array
	x = np.arange(0,int((mask.shape[0]/ncols)*ncols),ncols)
	y = np.arange(0,int((mask.shape[1]/nrows)*nrows),nrows)
	for i in x:
		mask[i-linewidth:i+linewidth,:]=0
	for j in y:
		mask[:,j-linewidth:j+linewidth]=0
	# mask out the edges of the FFT spectrum, the 'lowest' and 'highest' frequency x-y modes
	mask[0:2*linewidth,:]=0
	mask[mask.shape[0]-2*linewidth:mask.shape[0],:]=0
	mask[:,0:2*linewidth]=0
	mask[:,mask.shape[1]-2*linewidth:mask.shape[1]]=0
	#Remove any masking at the corners of the FFT where normal/non harmonic image features usually are 
	mask[0:2*recoverywidth,mask.shape[1]-recoverywidth:mask.shape[1]]=1
	mask[0:2*recoverywidth,0:recoverywidth]=1
	mask[mask.shape[0]-2*recoverywidth:mask.shape[0],mask.shape[1]-recoverywidth:mask.shape[1]]=1
	mask[mask.shape[0]-2*recoverywidth:mask.shape[0],0:recoverywidth]=1
	return mask

def fftfilter(image=None,linewidth=10,recoverywidth=100,gridshape=(511,256),shifted=True,savename=None,figsize=(15,20),wpad=4,titlefont=20,cbtickfont=20,axlabelfont=20,cbfont=20,cmap='Greys_r',nstd=1,dpi=600):
	'''Filters out periodic CMOS grid features from an optical image by finding suppressing them in the FFT spectrum.

	Keyword arguments:
	image (2D array) -- Image with grid features needing suppression
	linewidth (int) -- Along each harmonic of CMOS pattern, the FFT value is set to zero with a mask. High linewidth
						value improves suppression of less coherent harmonics but loses signal, and vice-versa.
	recoverywidth (int) -- Low frequency in FFT usally correlates with image signal you want. The low f region spanning 
							(dfx,dfy) = (0->2*recoverywidth, 0->recoverywidth) is not suppressed
	gridshape (tuple int) -- The 2D shape of the underlying CMOS array appearing in the optical image
	shifted (bool) -- If True the FFT spectrum is recentered around zero, else the spectrum is positive real valued.
	savename (str) -- Filename with absolut path to save file to. Use filetype extension .svg .

	Return arguments:
	mask (bool 2D array) -- A 2D binary mask, of same x,y shape as input 'images'. Corresponds to
						 FFT regions s.t. grid features are suppressed w.r.t. useful image signal. 
	'''
	mask = fft_mask(images=image, linewidth=linewidth, recoverywidth=recoverywidth) 
	fft = np.flip(fftpack.fft2(image),axis=1)
	fftmasked = np.flip(fft*mask,axis=1)
	if shifted: # plot the FFT spectrum recentered around zero
		fftabs = abs(np.fft.fftshift(fft,axes=(0,1))).astype('int')
		maskabs = abs(np.fft.fftshift(fftmasked,axes=(0,1))).astype('int')
	else: # plot the FFT spectrum as positive-real valued only
		fftabs = abs(fft).astype('int')
		maskabs = abs(fftmasked).astype('int')
	filtered_image = abs(fftpack.ifft2(fftmasked))#Invert FFT to recover image
	if savename!=None: # Plot max projection example, save to savepath
		fft_plot(absfft=fftabs, maskedfft=maskabs, nstd=nstd, savename=savename, 
			figsize=figsize,wpad=wpad,titlefont=titlefont,cbtickfont=cbtickfont,axlabelfont=axlabelfont,cbfont=cbfont,cmap=cmap,dpi=dpi)
	return filtered_image, fftabs, maskabs

#------------------------------------ Image and Timelapse Display Methods --------------------------------
def image_timelapse(manager,savename,data_times=None,data_names=None,vrange=[-4,1],normrows=[200,300],colormap='Blues',verbose=True,fps=10):
	'''Saves impedance data as a pure image timelapse, no labels/graphs etc.'''
	normrows=range(normrows[0],normrows[1])
	t0 = data_times[0]
	frames=[]
	for i in tqdm(range(len(data)), desc='-- Generating Timelapse --'):
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
		imprange=range(imrange[0],imrange[1],1) 
	myframes=[]
	for i in tqdm(imprange, desc ='-- Generating Timelapse --'):
		tx=timestamps[i]
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

def imp_display(image=None, std_range=(-4,2),vmin=None, vmax=None, imp_colormap='Greys', otsu_colormap='jet', nbins=255, nclass=3, figsize=(20,7),save=True,savename='imp_plot',dpi=600,alpha=0.5,rescale_factor=1e15):
	"""Display and save a single impedance image alongside it's histogram and a multi-class otsu segmentation result. Intended for quick overview."""
	image=deepcopy(image) # make sure not to modify original
	cmap_disc = cm.get_cmap(otsu_colormap, nclass) # Make a discreet n-class colormap
	n1,n2=std_range;
	if vmin == None:
		vmin=np.mean(image)+n1*np.std(image);
	if vmax == None:
		vmax=np.mean(image)+n2*np.std(image);
	thresholds = threshold_multiotsu(image,classes=nclass)
	otsumask = np.digitize(image, bins=thresholds)
	if np.mean(image)<(1/rescale_factor):
		image*=rescale_factor;vmin*=rescale_factor;vmax*=rescale_factor;thresholds*=rescale_factor;
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
	im=ax[0].imshow(image,vmin=vmin,vmax=vmax, cmap=imp_colormap)
	ax[0].set_title('Impedance Image')
	cb0=fig.colorbar(im,ax=ax[0],label='Capacitance [fFarads]')
	ax[1].hist(image.ravel(), bins=nbins)
	ax[1].set_title('Capacitance Spectrum')
	for thresh in thresholds:
		ax[1].axvline(thresh, color='r',label='otsu-thresh')
	ax[1].set_xlabel('Capacitance [fFarads]')
	ax[1].set_ylabel('Pixel Count')
	ax[1].axvline(vmin,label='vmin',color='orange')
	ax[1].axvline(vmax,label='vmax',color='green')
	ax[1].legend()
	im1=ax[2].imshow(otsumask, cmap=cmap_disc)
	ax[2].set_title('Multi-Otsu Result (n={} class)'.format(nclass))
	cb1=fig.colorbar(im1,ax=ax[2],label='classification #'.format(nclass),ticks=list(range(nclass)))
	cb1.ax.set_yticklabels(list(range(nclass)))
	print('imp_min=',vmin,'[Farad]', 'imp_max=',vmax,'[Farad]')
	if save:
		plt.savefig('{}.tif'.format(savename), transparent=True,dpi=dpi)
	return otsumask, thresholds, vmin, vmax

def ect_display(image=None, std_range=(-4,2),vmin=None, vmax=None, imp_colormap='Greys', otsu_colormap='jet', nbins=255, nclass=3, figsize=(20,7),save=True,savename='imp_plot',dpi=600,alpha=0.5):
	"""Display and save a single impedance image alongside it's histogram and a multi-class otsu segmentation result. Intended for quick overview."""
	image=deepcopy(image) # make sure not to modify original
	cmap_disc = cm.get_cmap(otsu_colormap, nclass) # Make a discreet n-class colormap
	n1,n2=std_range;
	if vmin == None:
		vmin=np.mean(image)+n1*np.std(image);
	if vmax == None:
		vmax=np.mean(image)+n2*np.std(image);
	thresholds = threshold_multiotsu(image,classes=nclass)
	otsumask = np.digitize(image, bins=thresholds)
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
	im=ax[0].imshow(image,vmin=vmin,vmax=vmax, cmap=imp_colormap)
	ax[0].set_title('ECT Image')
	cb0=fig.colorbar(im,ax=ax[0],label='Capacitance [Farads]')
	ax[1].hist(image.ravel(), bins=nbins)
	ax[1].set_title('Capacitance Spectrum')
	for thresh in thresholds:
		ax[1].axvline(thresh, color='r',label='otsu-thresh')
	ax[1].set_xlabel('Capacitance [Farads]')
	ax[1].set_ylabel('Pixel Count')
	ax[1].axvline(vmin,label='vmin',color='orange')
	ax[1].axvline(vmax,label='vmax',color='green')
	ax[1].legend()
	im1=ax[2].imshow(otsumask, cmap=cmap_disc)
	ax[2].set_title('Multi-Otsu Result (n={} class)'.format(nclass))
	cb1=fig.colorbar(im1,ax=ax[2],label='classification #'.format(nclass),ticks=list(range(nclass)))
	cb1.ax.set_yticklabels(list(range(nclass)))
	print('ect_min=',vmin,'[fFarad]', 'ect_max=',vmax,'[fFarad]')
	if save:
		plt.savefig('{}.tif'.format(savename), transparent=True,dpi=dpi)
	return otsumask, thresholds, vmin, vmax

def tiff_display(image=None, std_range=(-4,2),vmin=None, vmax=None, tiff_colormap='plasma', otsu_colormap='jet', nbins=255, nclass=3, figsize=(20,7),save=True,savename='imp_plot',dpi=600):
	"""Display and save a single tiff image alongside it's histogram and a multi-class otsu segmentation result. Intended for quick overview."""
	image=deepcopy(image) # make sure not to modify original
	n1,n2=std_range;
	if vmin == None:
		vmin=np.mean(image)+n1*np.std(image);
	if vmax == None:
		vmax=np.mean(image)+n2*np.std(image);
	thresholds = threshold_multiotsu(image,classes=nclass)
	otsumask = np.digitize(image, bins=thresholds)
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
	im=ax[0].imshow(image,vmin=vmin,vmax=vmax, cmap=tiff_colormap)
	ax[0].set_title('Tiff Image')
	cb0=fig.colorbar(im,ax=ax[0],label='Intensity [a.u.]')
	ax[1].hist(image.ravel(), bins=nbins)
	ax[1].set_title('Intensity Spectrum')
	for thresh in thresholds:
		ax[1].axvline(thresh, color='r',label='otsu-thresh')
	ax[1].set_xlabel('Intensity [a.u.]')
	ax[1].set_ylabel('Count')
	ax[1].axvline(vmin,label='vmin',color='orange')
	ax[1].axvline(vmax,label='vmax',color='green')
	ax[1].legend()
	im1=ax[2].imshow(otsumask, cmap=cm.get_cmap(otsu_colormap, nclass))
	ax[2].set_title('Multi-Otsu Result (n={} class)'.format(nclass))
	cb1=fig.colorbar(im1,ax=ax[2],label='classification #'.format(nclass),ticks=list(range(nclass)))
	cb1.ax.set_yticklabels(list(range(nclass)))
	print('vmin=',vmin,'[a.u]', 'vmax=',vmax,'[a.u]')
	if save:
		plt.savefig('{}.tif'.format(savename), transparent=True,dpi=dpi)
	return otsumask, thresholds, vmin, vmax

def peaks_timelapse(images=None, sigma=3, std_mod=0.1,edgekernel=1,margin=0):
	'''Masks the area behind the leading/strongest edge found along the row axis of each image in the sequence/timelapse.
	Used with pellicle experiments to measure displacement over time and growth curves.

	Keyword arguments:
	images (3D array) -- t-ordered image sequence of shape (nimages,nrow,ncol) 
	sigma (float) -- Strength of the gaussian_filter to be applied to the gradient of each image.
						Reduces likelihood of noise being mis-labeling as a signal peak. 
	std_mod (float) -- Weight filtering for peak prominence, i.e. a peak must be significant,
						(w.r.t the standard deviation) before it is considered a local maxima.
	edgekernel (int) -- Used to determine the edge position from the mean edge position of column chunks.
							Chunk widths are of size edgekernel, in pixels.
	margin (int) -- Used to give a row offset to the detected edge position.
						Improves cooperation with otsu method.

	Return arguments:
	peak_places (3D array) -- Boolean t-ordered array of shape (nimages,nrow,ncol).
								The sequence represents the calculated masks for each image.'''
	imagenum = images.shape[0]; rownum = images.shape[1];
	x = np.linspace(1,rownum,num=rownum,dtype=np.int32)
	deriv_arrays = np.gradient(images,x,axis=1)
	smooth = gaussian_filter(deriv_arrays,sigma=sigma)
	deriv_arrays /= np.linalg.norm(deriv_arrays)
	smooth /= np.linalg.norm(smooth)
	peak_places = np.zeros_like(images).astype('float')
	for i in tqdm(range(smooth.shape[0]), desc = ' -- Running Sharp Edge Segmentation -- '):
		globpeakindex=0
		for j in range(smooth.shape[2]):
			windowed_linecut = np.mean(smooth[i,:,j:j+edgekernel],axis=1)
			prom = std_mod*np.std(windowed_linecut)
			indices, _ = find_peaks(windowed_linecut, prominence=prom)
			linepeak = np.max(windowed_linecut[indices])
			peakindex= np.where(windowed_linecut==linepeak)[0][0]
			if peakindex>globpeakindex:
				globpeakindex=peakindex
		globpeakindex+=margin
		peak_places[i,:globpeakindex,:] = 1
	return peak_places

def mask_timelapse(manager=None,images=None,timestamps=None,imrange=None,savename=None,impcolormap='Greys',otsucolormap='jet',nclass=3,fps=6,save=False,verbose=False):
	'''Edge detection masking for pellicle experiments.'''
	if imrange==None:
		imprange=range(images.shape[0])
	else:
		imprange=range(imrange[0],imrange[1],1)   
	#n-region otsu masks
	otsumasks =  [otsumask for i in tqdm(imprange, desc=' -- Running Multi-Otsu Segmentation -- ') for otsumask,thresholds in [detect_otsu(images[i], nclass=nclass)]]
	if save:
		myframes=[] #graphs
		vmin = np.min(images); vmax = np.max(images); t0 = timestamps[0]; #Fix vmin/vmax across timelapse
		for i in tqdm(imprange, desc='-- Generating Animation --'):
			tx=timestamps[i]
			fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
			fig.suptitle(f'Impedance Image Classification, Time Elapsed: {tx-t0}')
			ax = axes.ravel()
			ax[0] = plt.subplot(1, 2, 1)
			ax[1] = plt.subplot(1, 2, 2)
			im1 = ax[0].imshow(images[i],vmin=vmin,vmax=vmax, cmap=impcolormap)
			ax[0].set_title('Impedance')
			ax[0].axis('off')
			cb1=fig.colorbar(im1,ax=ax[0],label='Capacitance [Farads]')
			im2 = ax[1].imshow(otsumasks[i],vmin=0,vmax=nclass-1, cmap=cm.get_cmap(otsucolormap, nclass))
			ax[1].set_title('Multi-Otsu')
			ax[1].axis('off')
			cb2=fig.colorbar(im2,ax=ax[1],label=f'n={nclass} class',ticks=list(range(nclass)))
			cb2.ax.set_yticklabels(list(range(nclass)))
			if verbose:
				plt.show()
			fig.canvas.draw()   # draw the canvas, cache the renderer
			im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			myframes.append(im)
			plt.close(fig)
		print(' -- Saving plots as .gif animation -- ')
		plotdir = os.path.join(manager.logdir,'plots')
		if(not os.path.exists(plotdir)):
			os.mkdir(plotdir)
		path =os.path.join(plotdir,savename+'.gif')
		imageio.mimsave(path,myframes, fps=fps)
		print(f' --  Animation saved as {path}  -- ')
	return np.array(otsumasks)

def fft_plot(absfft=None,maskedfft=None, nstd=1, savename=None, figsize=(15,20),wpad=4,titlefont=20,cbtickfont=20,axlabelfont=20,cbfont=20,cmap='Greys_r',dpi=600):
	'''Function for plotting before and after fftfiltering'''
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)
	font = font_manager.FontProperties(family='times new roman', size=cbfont)
	fig.tight_layout(pad=wpad)
	fftmin=np.min(absfft); fftmax=np.mean(absfft)+nstd*np.std(absfft);
	im1 = ax[0].imshow(absfft,vmin=fftmin,vmax=fftmax,cmap=cmap)
	im2 = ax[1].imshow(maskedfft,vmin=fftmin,vmax=fftmax,cmap=cmap)

	ims=(im1,im2); titles=('Normalized 2D-FFT Spectrum','Masked 2D-FFT Spectrum');
	for i in range(ax.shape[0]):
		ax[i].set_xlabel(r'$f_x\ [px]^{-1}$', fontsize=axlabelfont)
		ax[i].set_ylabel(r'$f_y\ [px]^{-1}$', fontsize=axlabelfont)
		ax[i].set_title(titles[i], fontsize=titlefont)
		cb=fig.colorbar(ims[i],ax=ax[i],label='Intensity [a.u.]',fraction=0.087,pad=0.04)
		cb.ax.tick_params(labelsize=cbtickfont) 
		cb.ax.yaxis.label.set_font_properties(font)
	if savename!=None:
		plt.savefig(savename, transparent=True, dpi=dpi, format='svg')
		print(f'-- FFT plot saved as {savename} --')
	plt.show()
	return fig
