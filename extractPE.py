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
        p[i+1] = p[i]*(1-alpha) + c[i]*alpha
    return p


sbjList = [1,3,4,5,6,7,8,9,11,13,14,15,17,18,19,20, 23,24,25,26,27,29]

colNames = ['sbjId','faceId','nameId','conflict','zRT','RT']
df_group = pd.read_csv('ISPC_behavior.csv', header=0, names=colNames)
df_gp_wCP = pd.DataFrame()
gp_alphas = pd.DataFrame()

SCNT=-1


for S in sbjList:
    SCNT = SCNT+1    
    df = df_group.loc[df_group.sbjId==S,:]
    
    # for later Design matrix construction
    idxC = df[(df.conflict==0) & (df.zRT.notnull())].index
    idxI = df[(df.conflict==1) & (df.zRT.notnull())].index
    RT = np.concatenate((np.array(df.loc[idxC,'RT']), np.array(df.loc[idxI,'RT'])), axis=0)
    constC = np.transpose(np.array([[1],[0],[0],[0]])*np.ones(len(idxC),dtype=float))
    constI = np.transpose(np.array([[1],[0],[0],[0]])*np.ones(len(idxI),dtype=float))    
    bestSSE = -1
    
    # figure out each face/name's conflict trial sequence ahead of time for the loop
    conflict_face = []
    conflict_name = []
    for ID in range(1,9,1):  # for faceid/nameid 1-8
        conflict_face.append(df.loc[df.faceId==ID, 'conflict'])
        conflict_name.append(df.loc[df.nameId==ID, 'conflict'])
        
    # Run exhaustive search for the best fitting alpha_face, alpha_name using RT data
    for alpha_face in np.arange(0.01, 1, 0.01):
        for alpha_name in np.arange(0.01, 1, 0.01):
        # plug in the alphas and derive CP_face and CP_name
            CP_face=[]
            CP_name=[]
            for i in range(0,8,1):  # 0-7 (8 unique ids)
                CP_face.append(conflictPred(conflict_face[i], alpha_face))
                CP_name.append(conflictPred(conflict_name[i], alpha_name))
            # put CP back to the original df
            for i in range(0,8,1):
                df.loc[df[df.faceId==i+1].index,'CP_face']=CP_face[i]
                df.loc[df[df.nameId==i+1].index,'CP_name']=CP_name[i]
            
            # Construct design matrix and calculate the Sum of squared errors (SSEs) 
            dmC = np.array(df.loc[idxC,['CP_face','CP_name']])        
            dmC = np.concatenate((dmC, constC), axis=1)
            dmI = df.loc[idxI,['CP_face','CP_name']]        
            dmI = np.concatenate((dmI, constI), axis=1)
            dmI = dmI[:,[3,4,5,0,1,2]]
            DM = np.concatenate((dmC, dmI),axis=0)
        
            diff =  RT - np.dot(np.dot(DM, np.linalg.pinv(DM)),RT)
            SSE = np.square(diff).sum()
            
            if (bestSSE <0)|(SSE < bestSSE):
                bestAlpha = [alpha_face, alpha_name]
                bestSSE = SSE
                df_wCP  = df    
    # end of a subject's modeling, put best-fitting CP back to group df and save the best-fitting alphas
    df_gp_wCP = pd.concat([df_gp_wCP, df_wCP], axis=0)
    gp_alphas.loc[SCNT, 'sbjId']=S
    gp_alphas.loc[SCNT, 'alpha_face']=bestAlpha[0]
    gp_alphas.loc[SCNT, 'alpha_name']=bestAlpha[1]

# end of all subjects' modeling
df_gp_wCP.to_pickle('df_wCP.pkl')
gp_alphas.to_pickle('df_alphas.pkl')

