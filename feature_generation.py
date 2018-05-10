# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wave, struct
import csv 

t = np.arange(0, 20, 0.1)
s = np.sin(t)
#plt.plot(t, s)

#csv =pd.read_csv('../BirdSpeciesClass/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC/train/cepst_conc_cepst_nips4b_birds_trainfile030.txt', sep="  ", header=None)
for totalcontrol in range(140,150):
    waveFile =wave.open('wav/'+str(totalcontrol)+'.wav', mode='r')

    #waveFile = wave.open('sine.wav', 'r')

    length = waveFile.getnframes()
    print('# of channels =', waveFile.getnchannels())
    print('# of frames = ', length);
    print('number of bytes = ', waveFile.getsampwidth())
         
    samples = np.zeros(length)
    for i in range(0,length):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        samples[i] = data[0];
    #print('data = ', data[0])

    #csv = pd.read_csv ('data.txt', sep="   ", header=None)
    #print('number of bytes = ', wvRead.getsampwidth())
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(samples)
#    plt.show()
    #
    plt.subplot(2, 1, 2)
    
    Pxx, freqs, bins, aa= plt.specgram(samples, NFFT=512, Fs=40000, noverlap=400)
    plt.show()
    fig.savefig("figs/"+str(totalcontrol)+".png")
#    plt.savefig("figs/"+str(totalcontrol)+".png")
#    print('nFreq ', len(freqs), ' nBins ', len(bins))

#    X = np.arange(-3, 4)
#    Y = np.arange(-3, 4)
#    X, Y = np.meshgrid(X, Y)
#    Z = np.exp(-(X*X + Y*Y)/10)
#    Z = Z/np.sum(Z)
#
#    tmpPxx = Pxx
#    nFreqs = len(freqs)
#    nBins = len(bins)
#    #for i in range(11, nFreqs-11):
#    #    for j in range(11, nBins-11):
#    for i in range(4, nFreqs-4):
#    #    print(' I am here', i)
#        for j in range(4, nBins-4):
#            tSum = 0
#            for r in range(0, 7):
#                for c in range(0, 7):
#                    tSum = tSum + Z[r][c]*Pxx[i-3+r][j-3+c]
#            tmpPxx[i][j]= tSum


    #Pxx =5.0*Pxx;
    #X = np.arange(0, nBins)
    #Y = np.arange(0, nFreqs)
    #X, Y = np.meshgrid(X, Y)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, Pxx, cmap='gray')
    #plt.show()

#    fig = plt.figure()
#    plt.imshow(Pxx, cmap = 'flag', aspect='auto', origin='lower')
    continue
    '''
    Local texture analysis
    '''
    j = 0
    syl = []
    finished = False
    npPxx = np.asarray(tmpPxx)
    dbPxx = 20*np.log10(npPxx)
    global_max = np.max(dbPxx)
    gmu = 30
    lmu = 30
    sylMin = global_max - gmu

    while finished == False:
        
        iterMax = np.max(dbPxx)
       
        if iterMax >= sylMin:
            iterMin = iterMax - lmu
            idx = np.argmax(dbPxx)
            row = int(idx/nBins)
            col = int(idx - row*nBins)
            print ('im ', iterMax , 'row ', row, 'col ', col)

            curCol = col
            if curCol <= 0:
                leftDone = True
                left = 0
            else:
                leftDone = False
                left = curCol
            
    #        Find left limit     
            curRow = row       
            while leftDone == False:
                curCol = curCol - 1
                if curCol <= 0:
                    curCol = 0
                    leftDone = True
                    left = 0
                    break
                    
                freqUp = curRow - 10
                freqDown = curRow + 10
                if freqUp < 0:
                    freqUp = 0
                if freqDown >= nFreqs:
                    freqDown = nFreqs - 1
    #            freqLet = np.zeros(freqDown - freqUp+1)
                letMax = -1000
                for ii in range(freqUp, freqDown+1):
                    if dbPxx[ii][curCol] > letMax:
                        if curCol >= 0:
                            rowIdx = ii
                            letMax = dbPxx[ii][curCol]
                
                if letMax < iterMin:
                    left = curCol + 1
                    leftDone = True
                else:
                    curRow = rowIdx
                    
                if curCol <= 0:
                    leftDone = True
                    left = 0
    # find right 
            curCol = col
            if curCol >= (nBins-1):
                rightDone = True
                rght = nBins - 1
            else:
                rightDone = False  
                right = curCol
                
            curRow = row  
            while rightDone == False:
                curCol = curCol + 1
                if curCol >= (nBins-1):
                    curCol = nBins-1
                    rightDone = True
                    right = nBins - 1
                    break
               
                freqUp = curRow - 10
                freqDown = curRow + 10
                if freqUp < 0:
                    freqUp = 0
                if freqDown >= nFreqs:
                    freqDown = nFreqs - 1
    #            freqLet = np.zeros(freqDown - freqUp+1)
                letMax = -1000
                for ii in range(freqUp, freqDown+1):
                    if dbPxx[ii][curCol] > letMax:
                        if curCol <= (nBins-1):
                            letMax = dbPxx[ii][curCol]
                            rowIdx = ii
                            
                if letMax < iterMin:
                    right = curCol - 1
                    rightDone = True            
                else:
                    curRow = rowIdx
                    
                if curCol >= (nBins-1):
                    rightDone = True
                    right = nBins-1
                    
            print('left =', left, 'right ',right)            
            if (leftDone == True) and (rightDone == True):
                syl.append([left, right])
                for rIdx in range(0, nFreqs):
                    for cIdx in range(left, right+1):
                        dbPxx[rIdx][cIdx] = sylMin - 100
        
        else:
            finished = True
          
    print ('Syl ', syl)        
    #print(np.min(Pxx))

    sylPxx = np.zeros((nFreqs, nBins))
    nSyls = len(syl)
    for i in range (0, nSyls ):
        for r in range(0, nFreqs):
            for c in range(syl[i][0], syl[i][1]):
                sylPxx[r][c]= Pxx[r][c]
    #
    #fig = plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.imshow(Pxx, cmap = 'flag', aspect='auto', origin='lower')    
    #plt.subplot(2, 1, 2)
    #plt.imshow(sylPxx, cmap = 'flag', aspect='auto', origin='lower')                         
                             
    """
            Can be improved here to find multiple syllables in the same time bins
    """
    # Find local texture features
    intMin = np.min(npPxx)
    intMax = np.max(npPxx)
    intRange = np.arange(0, intMax+2, 2)
    nSteps = len(intRange)
    print('nSteps =', nSteps)
    print('int max =', intMax, ' int min = ', intMin)
    nFeats = 5*4
    texture = np.zeros((nSyls, nFeats))

    print(' Number of syllables' , nSyls)
    for s in range(0, nSyls):
        width = (syl[s][1]-syl[s][0]+1)
        ws = syl[s][0]
        coarseSyl = np.zeros((nFreqs,  width ))
        for r in range(0, nFreqs):
            for c in range(0, width):
                coarseSyl[r][c]=np.round(npPxx[r][ws+c]/2)
        print('max num = ', np.max(coarseSyl))
        
        glcm0 = np.zeros((nSteps, nSteps)) 
        for r in range(0, nFreqs):
            for c in range(0, width-1):
                pp = int(coarseSyl[r][c])
                p10 = int(coarseSyl[r][c+1])
                glcm0[pp][p10]=glcm0[pp][p10]+1       
                     
        glcm45 = np.zeros((nSteps, nSteps)) 
        for r in range(0, nFreqs-1):
            for c in range(0, width-1):
                pp = int(coarseSyl[r][c])
                p10 = int(coarseSyl[r+1][c+1])
                glcm45[pp][p10]=glcm45[pp][p10]+1
                      
        glcm90 = np.zeros((nSteps, nSteps)) 
        for r in range(0, nFreqs-1):
            for c in range(0, width):
                pp = int(coarseSyl[r][c])
                p10 = int(coarseSyl[r+1][c])
                glcm90[pp][p10]=glcm90[pp][p10]+1
                      
        glcm135 = np.zeros((nSteps, nSteps)) 
        for r in range(0, nFreqs-1):
            for c in range(0, width-1):
                pp = int(coarseSyl[r][c])
                p10 = int(coarseSyl[r+1][c-1])
                glcm135[pp][p10]=glcm135[pp][p10]+1
                      
        glcm0 = glcm0/np.sum(glcm0)
        glcm45 = glcm45/np.sum(glcm45)
        glcm90 = glcm90/np.sum(glcm90)
        glcm135 = glcm135/np.sum(glcm135)
        
        logConst = 0.01
        
        texture[s][0] = np.sum(glcm0*glcm0)
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + r*glcm0[r][c]
        mean = mSum
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(r-mean)*glcm0[r][c]
        sigma2 = mSum
        if sigma2==0:
            sigma2=0.0001
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(c-mean)*glcm0[r][c]
        texture[s][1] = mSum/sigma2

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-c)*(r-c)*glcm0[r][c]
        texture[s][2] = mSum
               
        texture[s][3] = - np.sum(glcm0*np.log(logConst+np.abs(glcm0)))  

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + glcm0[r][c]/(1+(r-c)*(r-c))
                
        texture[s][4] = mSum

    # Direction 45
        texture[s][5] = np.sum(glcm45*glcm45)
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + r*glcm45[r][c]
        mean = mSum
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(r-mean)*glcm45[r][c]
        sigma2 = mSum
        if sigma2==0:
            sigma2=0.0001
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(c-mean)*glcm45[r][c]
        texture[s][6] = mSum/sigma2

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-c)*(r-c)*glcm45[r][c]
        texture[s][7] = mSum
               
        texture[s][8] = - np.sum(glcm45*np.log(logConst+np.abs(glcm45)))  

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + glcm45[r][c]/(1+(r-c)*(r-c))
                
        texture[s][9] = mSum

    # Direction 90
        texture[s][10] = np.sum(glcm90*glcm90)
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + r*glcm90[r][c]
        mean = mSum
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(r-mean)*glcm90[r][c]
        sigma2 = mSum
        if sigma2==0:
            sigma2=0.0001

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(c-mean)*glcm90[r][c]
        texture[s][11] = mSum/sigma2

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-c)*(r-c)*glcm90[r][c]
        texture[s][12] = mSum
               
        texture[s][13] = - np.sum(glcm90*np.log(logConst+np.abs(glcm90)))  

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + glcm90[r][c]/(1+(r-c)*(r-c))
                
        texture[s][14] = mSum
            
     # Direction 135
        texture[s][15] = np.sum(glcm135*glcm135)
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + r*glcm135[r][c]
        mean = mSum
        
        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(r-mean)*glcm135[r][c]
        sigma2 = mSum
        if sigma2==0:
            sigma2=0.0001

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-mean)*(c-mean)*glcm135[r][c]
        texture[s][16] = mSum/sigma2

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + (r-c)*(r-c)*glcm135[r][c]
        texture[s][17] = mSum
               
        texture[s][18] = - np.sum(glcm135*np.log(logConst+np.abs(glcm135)))  

        mSum = 0
        for r in range(0, nSteps):
            for c in range(0, nSteps):
                mSum = mSum + glcm135[r][c]/(1+(r-c)*(r-c))
                
        texture[s][19] = mSum   

    appendedList=np.zeros(20)
    f = open('features/'+str(totalcontrol)+'.csv','a')
    out=csv.writer(f,delimiter=',',quoting=csv.QUOTE_ALL)
    for frow in range(0,nSyls):
        for fcol in range(0,20):
            appendedList[fcol]=texture[frow][fcol]
            if fcol==19:
                out.writerow(appendedList)
    f.close()
                # appendedList[:]=[]     # not necessary but good practice
    #cmap=gnuplot



    #plt.subplot(4,1,4)
    #deng = np.diff(eng)
    #plt.plot(deng)
    #plt.show()
    #



