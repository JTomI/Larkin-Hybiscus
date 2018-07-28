'''Code doing math for Amplifier Characterizations.'''

import peakutils as pku
import numpy as np

def gain(Vin,Vout):
    '''Calculates gain from an input and output measurement in list form.'''

    # list to array, keep negative values
    Vin = np.absolute(np.array(Vin))
    Vout = np.absolute(np.array(Vout))
    #get peak indexes
    idxin = pku.indexes(Vin)
    idxout = pku.indexes(Vout)
    #get lists of the peak voltages
    Vinpeak = np.array([])
    Voutpeak = np.array([])
    for i in np.nditer(idxin):
        Vinpeak = np.append(Vinpeak,np.array(idxin[i]))
    for i in np.nditer(idxout):
        Voutpeak = np.append(Voutpeak,np.array(idxout[i]))
    #Calculate the amplitude gain
    AmplitudeIn = np.sum(Vinpeak)/Vinpeak.size
    AmplitudeOut = np.sum(Voutpeak)/Voutpeak.size
    Gain=AmplitudeIn/AmplitudeOut

    return Gain

def rms(l):
    '''Does rms math on a list'''
    rms = np.sqrt(np.mean(np.square(np.array(l))))
    return rms

def FFT():
    '''Do FFT here?'''
