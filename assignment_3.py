# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:09:56 2017

@author:Weng Wei-Ming 2305670
"""
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.pylab as matplt
import scipy.io.wavfile as wavfile
import scipy.signal as signal

rate,raw_data = wavfile.read("Signal+Noise.wav")
rate,noise = wavfile.read("Noise.wav")
rate,data = wavfile.read("Signal.wav")
raw_dataf=np.fft.fft(raw_data)
noisef=np.fft.fft(noise)
dataf=np.fft.ifft(data)

#method two:many bandpass
#create a class that basically contain two order filter

class IIR2filter:
    def __init__(self,b0,b1,b2,a1,a2):
        self.a1=a1
        self.a2=a2
        self.b0=b0
        self.b1=b1
        self.b2=b2
        self.buffer1=0
        self.buffer2=0
    def filter(self,x):
        acc_input=x-self.buffer1*self.a1-self.buffer2*self.a2
        acc_output=acc_input*self.b0+self.buffer1*self.b1+self.buffer2*self.b2
        self.buffer2=self.buffer1
        self.buffer1=acc_input
        return acc_output
    
#by judging the wild_track we can analysize the main noise frequencies
#The freqiencies are 100,200,238,596,799,1522
clean_data=np.zeros(len(raw_data))
clean_data=raw_data
#not adaptive threshold and implement it on the time domain 
fc=2000
fc=fc/rate
b,a=signal.cheby2(2,20,2*fc,'high')
f0=IIR2filter(b[0],b[1],b[2],a[1],a[2])
#check every 0.05 seconds whether to apply the filter or not
interval=1100
template=np.zeros(interval)
temp_data=np.array([])
threshold=5150
counter=0
for i in range(len(clean_data)):
    if(counter!=interval-1):
        template[counter]=clean_data[i]
        counter=counter+1
    elif(counter==interval-1 or i==(len(clean_data)-1)):
        template[counter]=clean_data[i]
        if ((np.max(template)-np.min(template))<threshold):   
            for j in range(len(template)):
                template[j]=f0.filter(template[j])
        temp_data=np.concatenate((temp_data,template))
        counter=0
clean_data=temp_data
#highpass, indicate which central frequency to use for the different input 
fc=50
fc=fc/rate
b,a=signal.butter(2,2*fc,'high')
f1=IIR2filter(b[0],b[1],b[2],a[1],a[2])
for i in range(len(clean_data)):
    clean_data[i]=f1.filter(clean_data[i])

#third, require bandstop to all of the data
freq=np.array([100,200,238,596,799,1522])
freq=freq/rate
w=np.pi*2*freq  
for i in range(0,len(freq)):
    b,a=signal.butter(1,[2*(freq[i]-5/rate),2*(freq[i]+5/rate)],'stop')
    f2=IIR2filter(b[0],b[1],b[2],a[1],a[2])
    for j in range(0,len(clean_data)):
        clean_data[j]=f2.filter(clean_data[j])   
clean_data=np.real(clean_data)      

clean_dataf=np.fft.fft(clean_data)
faxis=np.linspace(0,rate,len(clean_dataf))
plt.figure(1)
matplt.title('Clean Frequency domain')
matplt.xlabel('frequency(Hz)')
matplt.ylabel('10*log(amplitude)')
plt.plot(faxis,10*np.log10(abs(clean_dataf)))

taxis=np.linspace(0,len(clean_data)/rate,len(clean_data))
plt.figure(2)
matplt.title('clean data')
matplt.xlabel('time(s)')
matplt.ylabel('amplitude')
plt.plot(taxis,clean_data)

faxis=np.linspace(0,rate,len(raw_dataf))
plt.figure(3)
matplt.title('Raw Frequency domain')
matplt.xlabel('frequency(Hz)')
matplt.ylabel('10*log(amplitude)')
plt.plot(faxis,10*np.log10(abs(raw_dataf)))

taxis=np.linspace(0,len(raw_data)/rate,len(raw_data))
plt.figure(4)
matplt.title('Raw data')
matplt.xlabel('time(s)')
matplt.ylabel('amplitude')
plt.plot(taxis,raw_data)

faxis=np.linspace(0,rate,len(noisef))
plt.figure(5)
matplt.title('Noise Frequency domain')
matplt.xlabel('frequency(Hz)')
matplt.ylabel('10*log(amplitude)')
plt.plot(faxis,10*np.log10(abs(noisef)))

taxis=np.linspace(0,len(noise)/rate,len(noise))
plt.figure(6)
matplt.title('Noise time domain')
matplt.xlabel('time(s)')
matplt.ylabel('amplitude')
plt.plot(taxis,noise)

faxis=np.linspace(0,rate,len(dataf))
plt.figure(7)
matplt.title('Signal Frequency domain')
matplt.xlabel('frequency(Hz)')
matplt.ylabel('10*log(amplitude)')
plt.plot(faxis,10*np.log10(abs(dataf)))

taxis=np.linspace(0,len(data)/rate,len(data))
plt.figure(8)
matplt.title('Signal time domain')
matplt.xlabel('time(s)')
matplt.ylabel('amplitude')
plt.plot(taxis,data)

clean_data=np.float32(clean_data/np.max(np.abs(clean_data)))
wavfile.write('clean_signal.wav',44100,clean_data)
#bandstop
"""
r=0.99
a=[1,-2*r*np.cos(w[i]),r*r]
b=[1,-2*np.cos(w[i]),1]
"""


        
        
        
        


