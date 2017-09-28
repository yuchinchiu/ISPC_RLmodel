# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:35:15 2017

@author: yc180
"""

import os
import pandas as pd
import numpy as np

workingDir = os.path.dirname(os.path.realpath(__file__))
os.chdir(workingDir)

def conflictPred(c,alpha):
    # regurn trial-by-trial conflict prediction based on a given conflict sequence and learning rate- alpha
    c=np.array(c)
    p = np.zeros(len(c))+0.5
    for i in range(len(c)-1):
        p[i+1] = p[i]+ alpha*(c[i]-p[i])
    return p

colNames = ['sbjId','faceId','nameId','conflict','zRT','RT']
df_group = pd.read_csv('ISPC_behavior.csv', header=0, names=colNames)
df_gp_wCP = pd.DataFrame()
gp_alphas = pd.DataFrame()

SCNT=-1

for S in np.unique(df_group.sbjId):    
    df = df_group.loc[df_group.sbjId==S,:]    
    # for later Design matrix construction
    idxC = df[(df.conflict==0) & (df.zRT.notnull())].index
    idxI = df[(df.conflict==1) & (df.zRT.notnull())].index
    RT = np.concatenate((np.array(df.loc[idxC,'RT']), np.array(df.loc[idxI,'RT'])), axis=0)
    constC = np.transpose(np.array([[1],[0],[0],[0]])*np.ones(len(idxC),dtype=float))
    constI = np.transpose(np.array([[1],[0],[0],[0]])*np.ones(len(idxI),dtype=float))
    # Run exhaustive search for the best fitting alpha_face, alpha_name using RT data     
    bestSSE = -1        
    for alpha_face in np.arange(0.01, 1, 0.01):
        for alpha_name in np.arange(0.01, 1, 0.01):
        # plug in the alphas and derive CP_face and CP_name
            for i in range(0,8,1):  # 0-7 (8 unique ids)
                df.loc[df.faceId==i+1,'CP_face'] = conflictPred(df.loc[df.faceId==i+1,'conflict'], alpha_face)
                df.loc[df.nameId==i+1,'CP_name'] = conflictPred(df.loc[df.nameId==i+1,'conflict'], alpha_name)
            
            # Construct design matrix and calculate the Sum of squared errors (SSEs) 
            dmC = df.loc[idxC,['CP_face','CP_name']]
            dmC = np.concatenate((dmC, constC), axis=1)
            dmI = df.loc[idxI,['CP_face','CP_name']]        
            dmI = np.concatenate((dmI, constI), axis=1)
            dmI = dmI[:,[3,4,5,0,1,2]]
            DM = np.concatenate((dmC, dmI),axis=0)            
            diff =  RT - np.dot(np.dot(DM, np.linalg.pinv(DM)),RT)
            SSE = np.square(diff).sum()            
            # presever the best model
            if (bestSSE <0)|(SSE < bestSSE):
                bestAlpha = [alpha_face, alpha_name]
                bestSSE = SSE
                df_wCP  = df                
    # end of a subject's modeling, put best-fitting CP back to group df and save the best-fitting alphas
    df_gp_wCP = pd.concat([df_gp_wCP, df_wCP], axis=0)
    gp_alphas.loc[SCNT, 'sbjId']=S
    gp_alphas.loc[SCNT, 'alpha_face'] = bestAlpha[0]
    gp_alphas.loc[SCNT, 'alpha_name'] = bestAlpha[1]

# end of all subjects' modeling
df_gp_wCP.to_pickle('df_wCP_v2.pkl')
gp_alphas.to_pickle('df_alphas_v2.pkl')



