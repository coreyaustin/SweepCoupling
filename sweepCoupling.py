#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:47:03 2019

Code to calculate coupling functions from sweep data. Also find the best 
witness sensor using a broadband (or other) injection in a nearby (but 
different) location.

@author: coreyaustin
"""
#%%
from gwpy.timeseries import TimeSeriesDict
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

################################################################################
## Some helper functions for fetching and preparing data for use in the coupling
## functions calculations. Saves time series data in an .hdf5 file and saves
## spectrograms and spectra in a .npy file
################################################################################

# return a spectrogram and a normalized spectrogram 
# both are cropped in frequency to speed things up
def specgram(channel,fftl=4,ovlp=2):
    spec = channel.spectrogram2(fftlength=fftl,overlap=ovlp)**(1/2.)
    spec = spec.crop_frequencies(low=20,high=120)
    norm = spec.ratio('median')
    return spec,norm

#fetch data for given channels at a given time and save the data, then 
#calculate spectrogram and normalized spectrogram and save in a separate file
#return both spectrograms
def getData(channels,start,stop,filename):
    data = TimeSeriesDict.fetch(channels,start,stop)
    spec = {}
    for i in channels:
        spec[i] = {}
        spec[i]['sp'],spec[i]['norm'] = specgram(data[i])
        spec[i]['sp_asd'] = spec[i]['sp'].percentile(50)
    data.write('{}.hdf5'.format(filename),overwrite=True)
    np.save(filename,spec)
    return spec 

#load time series data stored in an .hdf5 file and return the spectrogram and
#normalized spectrogram
def loadHDF5(filename):
    data = TimeSeriesDict.read('{}.hdf5'.format(filename))
    spec = {}
    for i in channels:
        spec[i] = {}
        spec[i]['sp'],spec[i]['norm'] = specgram(data[i])
    np.save(filename,spec)
    return spec

#return contents of a numpy file (they contain spectrograms in this code, but 
#this should work for any numpy file)
def loadNPY(filename):
    return np.load(filename).item()

#return calibrated DARM in strain from a given DARM spectrum
def calDARM(self,darmasd,calfile='./data/L1darmcal_Apr17.txt'):
    caldarm = np.loadtxt(calfile)
    darmcal = interpolate.interp1d(caldarm[:,0],caldarm[:,1],
                fill_value='extrapolate')(darmasd.frequencies)
    darmasd *= 10**(darmcal/20)/4000
    return darmasd

###############################################################################
# Class for calculating coupling functions and finding the best sensor at each
# frequency.
###############################################################################
    
class SweepData:
    
    #Finds the frequency of the sweep at a given time by finding the highest
    #value (the brightest part of the spectrogram at a given time) in the
    #accelerometer at each time. Then averages together times where the 
    #sweep was passing through the same frequency.
    def averageASD(self,frequencies,channels):
        self.channels = channels
        for i in channels[1:]:
            sensor = self.data[i]
            sensor['darm'] = []
            sensor['mfreq'] = []
            sensor['avg'] = []
            len_time = len(sensor['norm'].xindex)
            for j in xrange(len_time):
                mval = np.argmax(sensor['norm'][j,:])
                mfreq = sensor['norm'].yindex[:][mval].value
                sensor['mfreq'].append(mfreq)
            for k in xrange(len(frequencies)):
                lowf  = frequencies[k] - 0.5
                highf = frequencies[k] + 0.25
                idx = np.where((sensor['mfreq']>lowf)&(sensor['mfreq']<highf))
                avg = np.mean(np.array(sensor['sp'][idx]),axis=0)
                sensor['avg'].append(avg)
                avg = np.mean(np.array(self.data['L1:GDS-CALIB_STRAIN']['sp'][idx]),axis=0)
                sensor['darm'].append(avg)
            
    #Calculate 'real' and upper limit coupling function and estimated ambient
    #for each accelerometer.
    def coupFunc(self,freqs):
        for i in self.channels[1:]:
            sensor = self.data[i]
            sensor['rfactor'] = []
            sensor['rfreq'] = []
            sensor['rest'] = []
            sensor['ufactor'] = []
            sensor['ufreq'] = []
            sensor['uest'] = []
            for j in xrange(len(freqs)):
                idx = np.where(sensor['sp'].yindex.value==freqs[j])[0][0]
                sens_inj = sensor['avg'][j][idx-1:idx+2]
                sens_bg  = self.quiet[i]['sp_asd'][idx-1:idx+2].value
                darm_inj = sensor['darm'][j][idx-1:idx+2]
                darm_bg  = self.quiet['L1:GDS-CALIB_STRAIN']['sp_asd'][idx-1:idx+2].value
                sens_rat = sens_inj[1]/sens_bg[1]
                darm_rat = darm_inj[1]/darm_bg[1]
                if sens_rat >= 5 and darm_rat >= 2:
                    darm = sum(darm_inj**2)-sum(darm_bg**2)
                    sens = sum(sens_inj**2)-sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor['rfactor'].append(factor)
                    sensor['rfreq'].append(freqs[j])
                    sensor['rest'].append(est)
                elif sens_rat >=5 and darm_rat < 2:      
                    darm = sum(darm_bg**2)
                    sens = sum(sens_inj**2) - sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor['ufactor'].append(factor)
                    sensor['ufreq'].append(freqs[j])
                    sensor['uest'].append(est) 
                    
#    def coupBest():
        

#%%

#channels  = ['L1:CAL-DELTAL_EXTERNAL_DQ','L1:PEM-CS_ACC_HAM5_SRM_Z_DQ','L1:PEM-CS_ACC_HAM6VAC_SEPTUM_X_DQ',
#             'L1:PEM-CS_ACC_HAM6VAC_SEPTUM_Y_DQ']
            
channels  = ['L1:GDS-CALIB_STRAIN','L1:PEM-CS_ACC_HAM5_SRM_Z_DQ','L1:PEM-CS_ACC_HAM6VAC_SEPTUM_X_DQ',
             'L1:PEM-CS_ACC_HAM6VAC_SEPTUM_Y_DQ']

#Start and end time for quiet references
q_start = 'Feb 15 2019 02:26:15 UTC'
q_end   = 'Feb 15 2019 02:27:15 UTC'

#Start and end time for broadband injection
i_start = 'Jan 14 2019 02:44:26 UTC'
i_end   = 'Jan 14 2019 02:45:26 UTC'

freqs = np.arange(31,91,0.5)

#%%

ham5sweep = SweepData()

#ham5sweep.data = loadHDF5('./data/190215_31to91_ham5_x_sweep')
ham5sweep.data = loadNPY('./data/190215_31to91_ham5_x_sweep.npy')

#ham5sweep.quiet = getData(channels,q_start,q_end,'./data/quiet')
ham5sweep.quiet = loadNPY('./data/quiet.npy')

ham5sweep.averageASD(freqs,channels)
ham5sweep.coupFunc(freqs)

#%%

limx   = [26,95]
limy   = [5e-25,6e-23]

for i in channels[1:]:
    f1, (ax1) = plt.subplots(1, sharex=False, figsize=[16,9])
    
    ax1.plot(ham5sweep.quiet['L1:GDS-CALIB_STRAIN']['sp_asd'],'steelblue',
             linewidth=2,label='DARM (quiet reference)')
    ax1.plot(ham5sweep.quiet['L1:GDS-CALIB_STRAIN']['sp_asd']/10,'--',
             c='lightsteelblue',label='DARM/10')
    ax1.scatter(ham5sweep.data[i]['rfreq'],ham5sweep.data[i]['rest'],'salmon',label=i[14:])
    
    ax1.set_yscale('log')
    ax1.set_xlim(limx)
#    ax1.set_ylim(limy)
    ax1.set_xlabel('Frequency (Hz)',color='dimgray',fontsize=14)
    ax1.set_ylabel(r'Strain/$\sqrt{Hz}$',color='dimgray',fontsize=14)
    ax1.set_title('Estimated Ambient ({})'.format(i[14:]),color='dimgray',fontsize=16)
    ax1.legend()
    ax1.grid(which='both',axis='both',color='darkgrey',linestyle='dotted')  
    ax1.tick_params(axis='both', colors='dimgrey', labelsize=14) 
    
#    plt.savefig('./plots/{}_coupling.png'.format(i))




