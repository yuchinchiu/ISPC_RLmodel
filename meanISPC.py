# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:42:51 2017

@author: yc180
"""

import os
import pandas as pd
import numpy as np

workingDir = os.path.dirname(os.path.realpath(__file__))
os.chdir(workingDir)


df_gp = pd.read_pickle('df_wCP.pkl')
sbjList = [1,3,4,5,6,7,8,9,11,13,14,15,17,18,18,29,23,24,25,26,27,29]

#sbjList = np.unique(df_gp.sbjId)

df_gp.loc[df_gp.faceId<=4,'bkType'] = 'SC'
df_gp.loc[df_gp.faceId>=5,'bkType'] = 'SR'
df_gp['bkType']  = pd.Categorical(df_gp.bkType, categories=['SC','SR'], ordered=True)


gp_ISPC = pd.DataFrame()
gp_ISPC['sbjId'] = sbjList

gp_meanRT= pd.DataFrame()

for S in sbjList: 
    df = df_gp[df_gp.sbjId==S]
    for bkType in [1,2]:
        if bkType==1:
            for f in range(1,5,1):
                df.loc[df.faceId==f,'conflictProb']= df[df.faceId==f].conflict.mean()
        else:
            for n in range(5,9,1):
                df.loc[df.nameId==n,'conflictProb']= df[df.nameId==n].conflict.mean()
    
    
    df.conflict.replace(0,'congruent', inplace=True)
    df.conflict.replace(1,'incongruent', inplace=True)
    df['conflict']  = pd.Categorical(df.conflict, categories=['congruent','incongruent'], ordered=True)
    df.conflictProb.replace(0.25,'rare_Inc', inplace=True)
    df.conflictProb.replace(0.75,'freq_Inc', inplace=True)
    df['conflictProb']  = pd.Categorical(df.conflictProb, categories=['rare_Inc','freq_Inc'], ordered=True)
    sbjRT = df.groupby(['bkType','conflictProb','conflict']).RT.mean()
    
    gp_ISPC.loc[gp_ISPC.sbjId==S,'SC_ISPC']=(sbjRT[1]-sbjRT[0])-(sbjRT[3]-sbjRT[2])
    gp_ISPC.loc[gp_ISPC.sbjId==S,'SR_ISPC']=(sbjRT[5]-sbjRT[4])-(sbjRT[7]-sbjRT[6])
    
    sbjRT = pd.DataFrame(sbjRT)
    sbjRT['sbjId']=S
    
    gp_meanRT = pd.concat((gp_meanRT, sbjRT),axis=0)



Weights = pd.read_pickle('gp_PEweights.pkl')
gp_meanRT.reset_index(inplace=True)
gp_ISPC = gp_ISPC.merge(Weights)



#%% Plot the mean ISPC as a function of 2 bkType x 2 ConflictProb x 2 trialType

#import matplotlib.pyplot as plt
import seaborn as sns
sns.factorplot(x='conflictProb',y = 'RT', data=gp_meanRT, hue='conflict',col='bkType')

#sns.regplot(x = 'SC_ISPC', y = 'SC_face', data = gp_ISPC,color="r")
#sns.regplot(x = 'SC_ISPC', y = 'SC_name', data = gp_ISPC,color="b")

#sns.regplot(x = 'SR_ISPC', y = 'SR_face', data = gp_ISPC,color="r")
#sns.regplot(x = 'SR_ISPC', y = 'SR_name', data = gp_ISPC,color="b")

