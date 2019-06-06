import zhinst.utils
from math import floor, ceil, pi, sqrt
import time 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class mfli(object):
	def __init__(self, apilevel = 6, device_id = 'dev4021'):
		self.apilevel, self.device_id = apilevel,device_id
		(self.daq, self.device, self.props) = zhinst.utils.create_api_session(device_id, apilevel)

	def lowpassBW(self,tc, order=1):
		fc = 1/(2*pi*tc)
		return fc*sqrt(2**(1.0/order)-1)

	def ziSweep(self,start = 1e4,stop = 5e6 ,samples = 1000 ,sigma = 1e-4,sweeporder = 4,averages = 1,input_range=1,Vrms_output=.01):
		#convert desired Vrms value to Vpeak-peak
		Vpp = Vrms_output*np.sqrt(2)
		#turn output on
		self.daq.setInt('/dev4021/sigouts/0/on', 1)
		self.daq.setInt('/dev4021/sigouts/0/enables/1', 1)
		self.daq.setDouble('/dev4021/sigins/0/range', input_range)
		self.daq.setDouble('/dev4021/sigouts/0/amplitudes/1', Vpp)
		
		h = self.daq.sweep()
		h.set('sweep/xmapping', 1)
		h.set('sweep/device', self.device_id)
		h.set('sweep/settling/inaccuracy', sigma)
		h.set('sweep/averaging/sample', averages)
		h.set('sweep/bandwidth', 1000.0000000)
		h.set('sweep/maxbandwidth', 1250000.0000000)
		h.set('sweep/order', sweeporder)
		h.set('sweep/gridnode', '/dev4021/oscs/0/freq')
		h.set('sweep/averaging/tc', 0.0000000)
		h.set('sweep/averaging/time', 0.0000000)
		h.set('sweep/bandwidth', 1000.0000000)
		h.set('sweep/omegasuppression', 40.0000000)
		h.set('sweep/start', start)
		h.set('sweep/stop',stop)
		h.set('sweep/samplecount', samples)
		h.set('sweep/endless', 0)
		h.subscribe('/dev4021/demods/0/sample')
		h.execute()
		result = 0
		p = 0
		while not h.finished():
			time.sleep(1);
			result = h.read()
			pcurr =float(h.progress())
			if (pcurr-p)>0.10:
				print "Progress %.2f%%\r" %(pcurr * 100)
				p = pcurr
		h.finish()
		h.unsubscribe('*')
		#turn output off
		self.daq.setInt('/dev4021/sigouts/0/on', 0)
		self.daq.setInt('/dev4021/sigouts/0/enables/1', 0)
		print('Finished\n')
		
		sampleno = result['dev4021']['demods']['0']['sample'][0][0]['samplecount']
		f = result['dev4021']['demods']['0']['sample'][0][0]['frequency']
		v = result['dev4021']['demods']['0']['sample'][0][0]['r']
		print("Frequency Range: [{} , {}]".format(float(start),float(stop)))
		print("Amplitude (V) at V+ signal output: {} rms , {} pp".format(Vrms_output,Vpp))
		print("Number of Samples: {}".format(float(sampleno)))
		print("Number of Averages: {}\n".format(float(averages)))
		return f, v
		
	def ziSpectrum(self,frequency, samplingRate = 8192, samples=1024, averages=1, filterOrder=3):
		self.daq.setDouble('/{}/oscs/0/freq'.format(self.device_id), frequency)
		self.daq.setDouble('/{}/demods/0/rate'.format(self.device_id), samplingRate)
		self.daq.setInt('/{}/demods/0/order'.format(self.device_id), filterOrder)
		#turn output off
		self.daq.setInt('/dev4021/sigouts/0/on', 0)
		self.daq.setInt('/dev4021/sigouts/0/enables/1', 0)
		
		Fs = self.daq.getDouble('/{}/demods/0/rate'.format(self.device_id))  # read the set sampling freq
		Fnyq = Fs/2  # nyquist frequency
		demod_tc = zhinst.utils.tc2bw(Fnyq/10.0, filterOrder)  # 3db point of demodulator
		self.daq.setDouble('/{}/demods/0/timeconstant'.format(self.device_id), demod_tc)
		
		time.sleep(10*demod_tc)
		
		
		h = self.daq.dataAcquisitionModule()
		h.set('dataAcquisitionModule/triggernode', '/dev4021/demods/0/sample.R')
		h.set('dataAcquisitionModule/preview', 1)
		h.set('dataAcquisitionModule/grid/cols', samples)
		h.set('dataAcquisitionModule/device', 'dev4021')
		h.set('dataAcquisitionModule/type', 0)
		h.set('dataAcquisitionModule/grid/repetitions', averages)
		h.set('dataAcquisitionModule/endless', 0)
		h.subscribe('/dev4021/demods/0/sample.xiy.fft.abs.avg')
		h.execute()
		result = 0
		p = 0
		while not h.finished():
			time.sleep(1);
			result = h.read()
			pcurr =float(h.progress())
			if (pcurr-p)>0.10:
				print "Progress %.2f%%\r" %(pcurr * 100)
				p = pcurr
		h.finish()
		h.unsubscribe('*')
		print('Finished\n')
		
		ts = result['dev4021']['demods']['0']['sample.xiy.fft.abs.avg'][0]['timestamp'][0]
		v = result['dev4021']['demods']['0']['sample.xiy.fft.abs.avg'][0]['value'][0]

		header = result['dev4021']['demods']['0']['sample.xiy.fft.abs.avg'][0]['header']

		nenbw = float(header['nenbw'])
		print("Normalized Noise Effective BW: {}".format(float(header['nenbw'])))

		df = float(header['gridcoldelta'])
		print("Frequency Spacing: {} Hz".format(float(header['gridcoldelta'])))
		print("Number of Samples: {}".format(float(header['gridcols'])))
		print("Number of Averages: {}".format(float(header['gridrepetitions'])))
		print("Sampling Rate: {} Sa/s \n".format(floor(float(header['gridcoldelta'])*(float(header['gridcols'])+1))))

		fmax = float(header['gridcols'])/2*(float(header['gridcoldelta']))

		f=np.linspace(-1*fmax, fmax, int(header['gridcols']))
		
		psd = (v/sqrt(df * nenbw))**2
		
		return f, v, psd
	
	def ziSweepPlot(self,freq_list,volt_list,title='',legend=['300k','77k','4.2k'],xlim=[0,5e6],ylim=[0,100e-3],auto = True):
		fig,ax = plt.subplots()
		for i in range(len(volt_list)):
			ax.plot(freq_list[i],volt_list[i])
		ax.set_xscale('log')
		if not auto:
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
		ax.set_title(title)
		ax.set_ylabel('Demod R, V')
		ax.set_xlabel('Frequency, Hz')
		ax.legend(legend,loc = 2)
		ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
		ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
		plt.show()

	def ziSpectrumPSDPlot(self,f, psd):
		plt.semilogy(f, psd, color='xkcd:coral')
		plt.xlabel('FFT Frequency, Hz')
		plt.ylabel('PSD, V^2/Hz')
		plt.grid(True, which='both', linestyle="--", linewidth=2.0)
		
	def ziSpectrumPlot(self,f, v):
		plt.semilogy(f, v, color='xkcd:azure')
		plt.xlabel('FFT Frequency, Hz')
		plt.ylabel('|FFT(X+iY|, V)')
		plt.grid(True, which='both', linestyle="--", linewidth=2.0)
		
if __name__=='__main__':
	pass