import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import glob
from datetime import datetime
from itertools import chain
import imageio
from MinervaManager import MinervaManager as MM


def normalize_by_channel(image):
    '''Normalize by channel'''
    ch0mean = np.mean(image[normrows, :32])
    for ch in range(8):
        image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] / np.mean(image[normrows, ch*32:(ch+1)*32]) * ch0mean
    image = np.abs(image)
    return image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def remove_outliers(data,Nstd=5):
    med=np.median(np.ravel(data))
    std=np.std(np.ravel(data))
    data[np.abs(data-med)>(Nstd*std)] = med
    return data

def ph_plot(manager=None,norm_rows=[200,300],stdrange=[-4,1],mycolormap='Blues'):
	'''Plots pH timelapse from Minerva Logfile.'''

	normrows=range(norm_rows)
	for lognum,fullname in enumerate(manager.logfiles):
    	list_all = Get_List(fullname,sortby='time')
    	print('\n\nall\n\n',list_all)
    	list_pH = Get_List(fullname,filterstring='pH')
    	print('\n\npH\n\n',list_pH)
	    if(lognum==0):
	        t0 = Get_Time(fullname,list_pH[0])
	        print('t0= ',t0)
    # plot images
    if(lognum==0):
        myframes=[]
        colonysizes=[[],]*4
        startindex=1
        endindex=len(list_all)
        image_1_ref=None

    for i in range(startindex,endindex,1):
        if 'pH_' in list_all[i]:        
            V_SW = Get_Attr(fullname,list_all[i],'V_SW')
            V_CM = Get_Attr(fullname,list_all[i],'V_CM')
            f_sw = Get_Attr(fullname,list_all[i],'f_sw')
            T_int = Get_Attr(fullname,list_all[i],'T_int')
            C_int = Get_Attr(fullname,list_all[i],'C_int')

            image_2d_ph1 = Get_Data(fullname,
                                    list_all[i],
                                    dataname='image_2d_ph1')
            image_2d_ph2 = Get_Data(fullname,
                                    list_all[i],
                                    dataname='image_2d_ph2')

            image_2d_ph1 = normalize_by_channel(image_2d_ph1)
            image_2d_ph2 = normalize_by_channel(image_2d_ph2)    
            # ~~~~~~~~~~~~~~~~~~
            # remove outliers
            image_2d_ph1 = remove_outliers(image_2d_ph1)
            image_2d_ph2 = remove_outliers(image_2d_ph2)    
            # ~~~~~~~~~~~~~~~~~~
            # re-normalize again
            image_2d_ph1 = normalize_by_channel(image_2d_ph1)
            image_2d_ph2 = normalize_by_channel(image_2d_ph2)    
            image_1 = image_2d_ph2
            if image_1_ref is None:
                image_1_ref = image_1
                continue
            tx = Get_Time(fullname,list_all[i])
            fig = plt.figure(figsize=(12,6))
            grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2)
            ax_main = fig.add_subplot(grid[:, :])
            
                
            im1 = ax_main.imshow(np.flip(np.transpose(image_1),axis=1), #-np.median(image_1)), # [50:100,:40]),
                                vmin=np.mean(image_1[normrows,:])+stdrange[0]*np.std(image_1[normrows,:]), 
                                vmax=np.mean(image_1[normrows,:])+stdrange[1]*np.std(image_1[normrows,:]), 
                                cmap=mycolormap)
            fig.colorbar(im1,ax=ax_main)
            ax_main.set_title(str(lognum) + '   ' + str(i) + '   ' + list_all[i] + ' time elapsed ' + str(tx-t0))
            
            plt.show()
            
            # add to frames for animation
            fig.canvas.draw()       # draw the canvas, cache the renderer
            im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            myframes.append(im)

            logdir = manager.logdir
            plotdir = os.path.join(logdir,'plots')
			if(not os.path.exists(plotdir)):
			    os.mkdir(plotdir)


