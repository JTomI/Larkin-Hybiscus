import labrad
import time
import numpy as np
import peakutils as pku
import matplotlib.pyplot as plt
import math
import sys
from tqdm import tqdm_notebook as tqdm

def f_gain(redp,fs,fe,n=1,V=0.01,N=500, delay=0.0):
	''':param redp: Pass the labrad connection variable of the red pitaya here. 
	:param fs:  Sweep start frequency in Hz 
	:param fe: Sweep end frequency in Hz 
	:param n: SMA output used for sweep
	:param V:  Signal amplitude, constant during sweep. Default; V=0.01 
	:param N: Number of samples to be taken. The sweep will take N samples at frequencies in the specified range that are equally spaced on a log scale. Default; N=50
	:param delaytime: forced delay time between frequency increments, used to allow signal to settle. Default; delaytime=0.0
	:returns: Sweeps frequency on sma output 1, and measures the gain of a device using the red pitaya sma input ports. Source=sma output 1, DUT input=sma input 1, DUT output=sma input 2
	:rtype: returns 2D numpy array. First row has the sampled frequencies, second row has the associated gain measurements.'''
	freq = np.geomspace(fs,fe,N)
	Vout=[]
	fgain = []
	srat=125000000.0
	redp.set_dec(1)
	dec=1
	timenow = str(time.strftime('%H:%M'))
	print ('Sweep start at %(timenow)s' %{'timenow':timenow})
	for f in tqdm(np.nditer(freq,flags=['refs_ok']),desc='Freq Sweep'):
	    redp.func_gen(n,'sin',f,V)
	    time.sleep(delay)
	    Ns=int(buff_len(f,SRAT=(srat/dec)))
	    Vin = redp.acquire(1,Ns)
	    Vout = redp.acquire(2,Ns)
	    fgain.append(gain(Vin,Vout))
	print 'sweep done'
	return np.array([freq,fgain])

def v_gain(redp,Vs,Ve,f=100000,N=500,delay=0.0):
    '''
	:param redp: Pass the labrad connection variable of the red pitaya here.
	:param Vs:  Sweep start voltage in V
	:param Ve: Sweep end voltage in V
	:param V:  Signal amplitude, constant during sweep. Default; V=0.01
	:param N: Number of samples to be taken. The sweep will take N samples at frequencies in the specified range that are equally spaced on a log scale. Default; N=50
	:param delaytime: forced delay time between frequency increments, used to allow signal to settle. Default; delaytime=0.0 
	:returns: Sweeps signal amplitude on sma output 1, and measures the gain of a device using the red pitaya sma input ports. Source=sma output 1, DUT input=sma input 1, DUT output=sma input 2
	:rtype: returns 2D numpy array. First row has the sampled input voltages, second row has the associated gain measurements.  
	'''
    volt = np.linspace(Vs,Ve,num=N)
    vgain = []
    timenow = str(time.strftime('%H:%M'))
    print ('Sweep start at %(timenow)s, expect %(delay)fs delay.' %{'delay':N*.75,'timenow':timenow})
    for v in np.nditer(volt):
        redp.func_gen(1,'sin',f,float(v))
        #settle the signal
        dec=good_dec(f)
        redp.acq_dec(dec)
        time.sleep(delay)
        Vin = redp.acquire(1,16384)
        Vout = redp.acquire(2,16384)
        vgain.append(gain(Vin,Vout))
    print 'sweep done'
    return np.array([volt,vgain])

def save(numpyarray,name,bias):
	'''
	:param numpyarray: data to be saved
	:param name: Name the measurement, usually just sweep type
	:param bias: Transistor quiescent DC bias
	:returns: Saves array as txt file in current working directory
	'''
	desc = str(raw_input('Include brief measurement description here: \n'))
	date = str(time.strftime('%d:%m:%Y'))
	filedate = date.replace(':','_')
	timenow = str(time.strftime('%H:%M'))
	bias=float(bias)
	name=''.join([name,'_',filedate])
	np.savetxt(name,numpyarray, header=('Session save date %(date)s \n Session save time: %(time)s \n Quiescent DC bias: %(bias)f \n Measurement description: %(desc)s \n') %{'date':date,'time':timenow,'bias':bias,'desc':desc})
	return ('Saved array as %s' %name)

def rp_out(redp,fs,fe,V,nin=1,nout=1,N=500,delay=0.0):
	'''
	:param nout: sma output where generate function
	:param nin: sma input where want to listen to generated function
	:param fs: output sweep start Hz
	:param fs: output sweep end Hz
	:param V: output amplitude under test
	:returns: 2D Array of sourced frequency and actual voltage output.
	'''
	freq = np.geomspace(fs,fe,N)
	Vout=[]
	srat=125000000.0
	redp.set_dec(1)
	dec=1
	timenow = str(time.strftime('%H:%M'))
	print ('Sweep start at %(timenow)s' %{'timenow':timenow})
	for f in tqdm(np.nditer(freq,flags=['refs_ok']),desc='Output Sweep'):
		redp.func_gen(nout,'sin',float(f),V)
		time.sleep(delay)
		Ns=int(buff_len(f,SRAT=(srat/dec)))
		data = redp.acquire(nin,Ns)
		data = np.absolute(np.array(data))
		idxout = pku.indexes(data)
		Voutpeak = np.array([])
		for index in np.nditer(idxout, flags=['zerosize_ok']):
			Voutpeak = np.append(Voutpeak,np.array(data[index]))
		AmplitudeOut = np.sum(Voutpeak)/Voutpeak.size
		Vout.append(AmplitudeOut)
	print 'sweep done'
	return np.array([freq,Vout])

#------------------------------------------------------Math Below -------------------------------------------------------------------------------------------------------------------------------
def buff_len(f,SRAT=125000000.0,N=50.0,abso=True):
	'''Figures out how many samples to take at a given frequency for the 125MHz Sample rate, optionally with decmiation, to capture N peaks.'''
	if abso==True:
		Ntime=N*(1.0/(2*f)) # are you taking absolute value or not? gain amplitude info twice as fast if you do.
		
	else:
		Ntime=N*(1.0/f)
	return int(Ntime*SRAT)
	

def maxf(numpyarray):
	'''returns the x axis position of y axis maximum in array'''
	idx = pku.indexes(numpyarray)
	maxpk=np.amax(g)
	for i in idx:
	    if g[i]==maxpk:
	        maxf=f[i]
	        break
	    else:
	        pass
	return maxf

def gain(Vin,Vout):
    '''
    :param Vin: Input signal from DUT to be processed. Should be given as a list of floats.\n 
    :param Vout: Output signal from DUT to be processed. Should be given as a list of floats.\n
    :Returns: Calculates dB gain/attenuation from two lists of equal length, representing an input and output signal from a DUT.
    '''

    # list to array, keep negative values
    Vin = np.absolute(np.array(Vin))
    Vout = np.absolute(np.array(Vout))
    #get peak indexes
    idxin = pku.indexes(Vin,thres=.9)
    idxout = pku.indexes(Vout,thres=.9)
    #get lists of the peak voltages
    Vinpeak = np.array([])
    Voutpeak = np.array([])
    for index in np.nditer(idxin, flags=['zerosize_ok']):
        Vinpeak = np.append(Vinpeak,np.array(Vin[index]))
    for index in np.nditer(idxout, flags=['zerosize_ok']):
        Voutpeak = np.append(Voutpeak,np.array(Vout[index]))
    #Calculate the amplitude gain
    AmplitudeIn = np.sum(Vinpeak)/Vinpeak.size
    AmplitudeOut = np.sum(Voutpeak)/Voutpeak.size
    Gain=float(AmplitudeOut/AmplitudeIn)
    GaindB=20*math.log10(Gain)
    # return GaindB
    return Gain

def rms(lst):
    '''
    Takes in list, returns rms math of list values.
    '''
    rms = np.sqrt(np.mean(np.square(np.array(lst))))
    return rms

def good_dec(f):
	'''Calculates what is the best decimation factor to use for capturing peaks of a given  periodic signal frequency in one buffer of samples.'''
	decimation=[1,8,64,1024,8192,65536]
	fullbuffer=16384.0
	samplerate=125000000.0
	peakscaptured=[]
	if samplerate >= 2*f:
		for dec in decimation:
			if float(samplerate/dec) > 2*f:
				#determine # of peaks captured at each dec
				timetofill=float(fullbuffer/(samplerate/dec))
				peakscaptured.append(2*f*timetofill)
			else:
				pass
		idx=peakscaptured.index(max(peakscaptured))
		goodec=decimation[idx]
		return goodec
	else:
		print 'This frequency cannot be captured by the red pitaya'


#------------------------------------------------------Jupyter plotting below-------------------------------------------------------------------------------------------------------------------------------
#howto loadfiles
	# a,b=np.loadtxt('fsweep/fsweep_bias4_03_08_2018.txt')
	# c,d=np.loadtxt('fsweep/fsweep_bias5_03_08_2018')
	# e,f=np.loadtxt('fsweep/fsweep_bias6_03_08_2018')
	# g,h=np.loadtxt('fsweep/fsweep_bias7_03_08_2018')

#subplots
	# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	# plt.suptitle('Frequency sweeps at different DC bias ',fontsize=18,color='black')
	# fig.text(0.5, 0.01, 'Frequency (Hz)', ha='center', fontsize= 16)
	# fig.text(0.03, 0.5, 'Gain (dB)', va='center', rotation='vertical',fontsize= 16)
	# axes[0, 0].semilogx(a, b,'r.')
	# axes[0, 0].axhline(color='black')
	# axes[0, 0].set_title('4.0000V bias', fontsize=12)
	# axes[0, 1].semilogx(c, d,'r.')
	# axes[0, 1].axhline(color='black')
	# axes[0, 1].set_title('5.0000V bias', fontsize=12)
	# axes[1, 0].semilogx(e, f,'r.')
	# axes[1, 0].axhline(color='black')
	# axes[1, 0].set_title('6.0000V bias', fontsize=12)
	# axes[1, 1].semilogx(g, h,'r.')
	# axes[1, 1].axhline(color='black')
	# axes[1, 1].set_title('7.0000V bias', fontsize=12)
