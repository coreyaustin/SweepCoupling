#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:47:03 2019

@author: coreyaustin
"""
#%%

from gwpy.timeseries import TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

class SweepData:
    
    fftl=4
    ovlp=2
    
    def specgram(self,channel):
        spec = channel.spectrogram2(fftlength=self.fftl,overlap=self.ovlp)**(1/2.)
        spec = spec.crop_frequencies(low=20,high=120)
        norm = spec.ratio('median')
        return spec,norm
    
    def getData(self,channels,start,stop,filename,save=False,spASD=False):
        data   = TimeSeriesDict.fetch(channels,start,stop)
        for i in channels:
            data[i].sp,data[i].norm = self.specgram(data[i])
            if spASD:
                data[i].sp_asd = data[i].sp.percentile(50)
        if save:
            np.save(filename,data)
        return data            
        
    def loadHDF5(self,filename):
        data = TimeSeriesDict.read(filename)
        channels = data.keys()
        for i in channels:
            data[i].sp,data[i].norm = self.specgram(data[i])
        return data
    
    def loadNPY(self,filename):
        return np.load(filename).item()

    def calDARM(self,darmasd,calfile='./L1darmcal_Apr17.txt'):
        caldarm = np.loadtxt(calfile)
        darmcal = interpolate.interp1d(caldarm[:,0],caldarm[:,1],
                    fill_value='extrapolate')(self.quiet[channels[0]].sp_asd.frequencies)
        darmasd *= 10**(darmcal/20)/4000
        
    def getQuiet(self,channels,start,stop,filename):
        self.quiet = TimeSeriesDict.fetch(channels,start,stop)
        for i in channels:
            self.quiet[i].sp,self.quiet[i].norm = self.specgram(self.quiet[i])
            self.quiet[i].sp_asd = self.quiet[i].sp.percentile(50)
#            self.quiet[i].sp_asd = self.quiet[i].asd(4,2)
            self.quiet[i].sp_asd = self.quiet[i].sp_asd.crop(20,120)
#        self.calDARM(self.quiet[channels[0]].sp_asd)
        np.save(filename,self.quiet)
        

        
    def averageASD(self,frequencies):
        for i in self.channels[1:]:
            sensor = self.data[i]
            sensor.darm = []
            sensor.mfreq = []
            sensor.avg = []
            len_time = len(sensor.norm.xindex)
            for j in xrange(len_time):
                mval = np.argmax(sensor.norm[j,:])
                mfreq = sensor.norm.yindex[:][mval].value
                sensor.mfreq.append(mfreq)
            for k in xrange(len(frequencies)):
                lowf  = frequencies[k] - 0.5
                highf = frequencies[k] + 0.25
                idx = np.where((sensor.mfreq>lowf)&(sensor.mfreq<highf))
                avg = np.mean(np.array(sensor.sp[idx]),axis=0)
                sensor.avg.append(avg)
                avg = np.mean(np.array(self.data['L1:GDS-CALIB_STRAIN'].sp[idx]),axis=0)
                sensor.darm.append(avg)
            
    def coupFunc(self,freqs):
        for i in self.channels[1:]:
            sensor = self.data[i]
            sensor.rfactor = []
            sensor.rfreq = []
            sensor.rest = []
            sensor.ufactor = []
            sensor.ufreq = []
            sensor.uest = []
            for j in xrange(len(freqs)):
                idx = np.where(sensor.sp.yindex.value==freqs[j])[0][0]
                sens_inj = sensor.avg[j][idx-1:idx+2]
                sens_bg  = self.quiet[i].sp_asd[idx-1:idx+2].value
                darm_inj = sensor.darm[j][idx-1:idx+2]
                darm_bg  = self.quiet['L1:GDS-CALIB_STRAIN'].sp_asd[idx-1:idx+2].value
                sens_rat = sens_inj[1]/sens_bg[1]
                darm_rat = darm_inj[1]/darm_bg[1]
                if sens_rat >= 5 and darm_rat >= 2:
                    darm = sum(darm_inj**2)-sum(darm_bg**2)
                    sens = sum(sens_inj**2)-sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor.rfactor.append(factor)
                    sensor.rfreq.append(freqs[j])
                    sensor.rest.append(est)
                elif sens_rat >=5 and darm_rat < 2:      
                    darm = sum(darm_bg**2)
                    sens = sum(sens_inj**2) - sum(sens_bg**2)
                    factor = np.sqrt(darm/sens)
                    est    = factor * sens_bg[1]
                    sensor.ufactor.append(factor)
                    sensor.ufreq.append(freqs[j])
                    sensor.uest.append(est) 
                    
#    def coupBest():
        

#%%

#channels  = ['L1:CAL-DELTAL_EXTERNAL_DQ','L1:PEM-CS_ACC_HAM5_SRM_Z_DQ','L1:PEM-CS_ACC_HAM6VAC_SEPTUM_X_DQ',
#             'L1:PEM-CS_ACC_HAM6VAC_SEPTUM_Y_DQ']
            
channels  = ['L1:GDS-CALIB_STRAIN','L1:PEM-CS_ACC_HAM5_SRM_Z_DQ','L1:PEM-CS_ACC_HAM6VAC_SEPTUM_X_DQ',
             'L1:PEM-CS_ACC_HAM6VAC_SEPTUM_Y_DQ']

q_start = 'Feb 15 2019 02:26:15 UTC'
q_end   = 'Feb 15 2019 02:27:15 UTC'

freqs = np.arange(31,91,0.5)

#%%

ham5sweep = SweepData()
ham5sweep.dloadData('./data/190215_31to91_ham5_x_sweep.hdf5')
ham5sweep.averageASD(freqs)
#ham5sweep.getQuiet(channels,q_start,q_end,'fish')
ham5sweep.loadQuiet('fish.npy')
ham5sweep.coupFunc(freqs)

#%%

limx   = [26,95]
limy   = [5e-25,6e-23]

for i in channels[1:]:
    f1, (ax1) = plt.subplots(1, sharex=False, figsize=[16,9])
    
    ax1.plot(ham5sweep.quiet['L1:GDS-CALIB_STRAIN'].sp_asd,'steelblue',
             linewidth=2,label='DARM (quiet reference)')
    ax1.plot(ham5sweep.quiet['L1:GDS-CALIB_STRAIN'].sp_asd/10,'--',
             c='lightsteelblue',label='DARM/10')
    ax1.scatter(ham5sweep.data[i].rfreq,ham5sweep.data[i].rest,'salmon',label=i)
    
    ax1.set_yscale('log')
    ax1.set_xlim(limx)
#    ax1.set_ylim(limy)
    ax1.set_xlabel('Frequency (Hz)',color='dimgray',fontsize=14)
    ax1.set_ylabel(r'Strain/$\sqrt{Hz}$',color='dimgray',fontsize=14)
    ax1.set_title('Estimated Ambient ({})'.format(i),color='dimgray',fontsize=16)
    ax1.legend()
    ax1.grid(which='both',axis='both',color='darkgrey',linestyle='dotted')  
    ax1.tick_params(axis='both', colors='dimgrey', labelsize=14) 
    
#    plt.savefig('./plots/{}_coupling.png'.format(i))






