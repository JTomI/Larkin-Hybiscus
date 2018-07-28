import labrad
import CryoAmpMath as cma
import time


def rp_connection():
    '''redpitaya connections to servers handled here'''
    cxn = labrad.connect()
    rp = cxn.red_pit
    rp.select_device()
    return rp

def f_gain(V,fs,fe,df):
    '''Records gain over a frequency sweep at some voltage amplitude, on output 1'''
    #setup the test amplitude and frequencis in sweep.
    freq = range(fs-df,fe,df)
    gain = []
    for i in freq:
        redp.func_gen(1,'sin',freq[i],V)
        #settle the signal
        time.sleep(.01)
        Vin = redp.acquire(1,16384)
        Vout = redp.acquire(2,16384)
        gain = gain.append(cma.gain(Vin,Vout))
    #Try outputting 2 row numpy array, freq and gain
    return gain

def v_gain(f,Vs,Ve,dV):
    '''Records gain over a Voltage sweep on output 1'''
    volt = range(Vs-dV,Ve,dV)
    gain = []
    for i in volt:
        redp.func_gen(1,'sin',f,volt[i])
        #settle the signal
        time.sleep(.01)
        Vin = redp.acquire(1,16384)
        Vout = redp.acquire(2,16384)
        gain = gain.append(cma.gain(Vin,Vout))
    #Try outputting 2 row numpy array, volt and gain
    return gain

def thermometer(n,R,t):
    '''Reads noise on input, determines temperature for given resistance value'''
    noise = redp.acquire(n,t/125000000)
    Vrms=cma.rms(noise)
    Temperature =

if __name__ == '__main__':
    redp = rp_connection()
