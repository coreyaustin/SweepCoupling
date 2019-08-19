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
import numpy as np
from scipy import interpolate


#%%

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
        spec[i]['sp_asd'] = spec[i]['sp'].percentile(50)
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
                    darm = np.sum(darm_inj**2)-np.sum(darm_bg**2)
                    sens = np.sum(sens_inj**2)-np.sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor['rfactor'].append(factor)
                    sensor['rfreq'].append(freqs[j])
                    sensor['rest'].append(est)
                elif sens_rat >=5 and darm_rat < 2:      
                    darm = np.sum(darm_bg**2)
                    sens = np.sum(sens_inj**2) - np.sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor['ufactor'].append(factor)
                    sensor['ufreq'].append(freqs[j])
                    sensor['uest'].append(est) 
                    
    #find the best coupling function at each frequency by multiplying the coupling
    #function for each acclerometer by that accelerometer during a broadband 
    #injection in a nearby but different location and comparing the result to 
    #DARM during the injection               
    def coupBest(self):
        for i in self.channels[1:]:
            sensor = self.data[i]
            sensor['diff'] = []
            sensor['bbest'] = []
            for j in xrange(len(sensor['rfreq'])):
                sens_bband = self.bband[i]['sp_asd']
                darm_bband = self.bband[self.channels[0]]['sp_asd']
                bb_sens_idx = np.where(sens_bband.frequencies.value==sensor['rfreq'][j])
                bb_darm_idx = np.where(darm_bband.frequencies.value==sensor['rfreq'][j])
                sensor['diff'].append(((darm_bband[bb_darm_idx]-sensor['rfactor'][j]*
                                        sens_bband[bb_sens_idx])**2)**(1/2.))
        self.best = {}
        self.best['frequencies'] = np.unique(np.concatenate([self.data[k]['rfreq'] for k in self.channels[1:]]))
        self.best['sensor'] = []
        self.best['factor'] = []
        self.best['estimate'] = []
        for i in xrange(len(self.best['frequencies'])):
            diff = []
            sens = []
            fact = []
            est  = []
            for j in self.channels[1:]:
                sensor = self.data[j]
                idx = np.where(sensor['rfreq']==self.best['frequencies'][i])[0]
                if idx.size:
                    diff.append(sensor['diff'][idx[0]])
                    sens.append(j)
                    fact.append(sensor['rfactor'][idx[0]])
                    est.append(sensor['rest'][idx[0]])
            self.best['sensor'].append(sens[diff.index(min(diff))])
            self.best['factor'].append(fact[diff.index(min(diff))])
            self.best['estimate'].append(est[diff.index(min(diff))])
        self.best['estimate'] = np.array(self.best['estimate'])
        



