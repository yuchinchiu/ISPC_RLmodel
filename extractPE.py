# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:35:15 2017

@author: yc180
"""

import os
import pandas as pd
import numpy as np

#workingDir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(workingDir)

def conflictPred(c,alpha):
    # regurn trial-by-trial conflict prediction based on a given conflict sequence and learning rate- alpha
    c=np.array(c)
    p = np.zeros(len(c))+0.5
    for i in range(len(c)-1):
        p[i+1] = p[i]*(1-alpha) + c[i]*alpha
    return p


colNames = ['sbjId','faceId','nameId','conflict','zRT','RT']
df_group = pd.read_csv('ISPC_behavior.csv', header=0, names=colNames)
df_gp_wCP = pd.DataFrame()

gp_alphas=pd.DataFrame()


S = 1
SCNT=0
df = df_group.loc[df_group.sbjId==S,:]
idxC = df[(df.conflict==0) & (df.zRT.notnull())].index
idxI = df[(df.conflict==1) & (df.zRT.notnull())].index
bestSSE = -1

for alpha_face in np.arange(0.01,1,0.01):
    for alpha_name in np.arange(0.01,1,0.01):
        
        # plug in the alphas and derive CP_face and CP_name
        for f in np.unique(df.faceId):
            c = df.loc[df.faceId==f, 'conflict']
            p = conflictPred(c, alpha_face)
            df.loc[df.faceId==f, 'CP_face'] = p
            
        for n in np.unique(df.nameId):
            c = df.loc[df.nameId==n, 'conflict']
            p = conflictPred(c, alpha_name)
            df.loc[df.nameId==n, 'CP_name'] = p            
            
        # Construct design matrix and calculate the Sum of squared errors (SSEs) 
        dmC = df.loc[idxC,['CP_face','CP_name']]
        dmC['constant'] = 1
        dmI = df.loc[idxI,['CP_face','CP_name']]
        dmI['constant'] = 1        
        dmC = np.array(dmC)
        dmI = np.array(dmI)
        RT1 = np.array(df.loc[idxC,'RT'])
        diffC =  RT1 - np.dot(np.dot(dmC, np.linalg.pinv(dmC)),RT1)
        RT2 = np.array(df.loc[idxI,'RT'])
        diffI =  RT2 - np.dot(np.dot(dmI, np.linalg.pinv(dmI)),RT2)
        SSE = np.square(diffC).sum() + np.square(diffI).sum()
        
        if (bestSSE <0)|(SSE < bestSSE):
            bestAlpha = [alpha_face, alpha_name]
            bestSSE = SSE
            df_wCP  = df    
        # end of a subject's modeling
        
        
df_gp_wCP = pd.concat([df_gp_wCP, df_wCP], axis=0)
gp_alphas.loc[SCNT, 'sbjId']=S
gp_alphas.loc[SCNT, 'alpha_face']=bestAlpha[0]
gp_alphas.loc[SCNT, 'alpha_name']=bestAlpha[1]

#%%



