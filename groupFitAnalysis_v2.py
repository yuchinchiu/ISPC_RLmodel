# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:45:05 2017

@author: yc180
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


workingDir = os.path.dirname(os.path.realpath(__file__))
os.chdir(workingDir)


df_gp = pd.read_pickle('df_wCP_v3.pkl')
#sbjList = np.unique(df_gp.sbjId)
sbjList = [1,3,4,5,6,7,8,9,11,13,14,15,17,18,19,20,23,24,25,26,27,29]

newCol = ['bkType','runId','faceRep','nameRep','fullRepAlt','PE_face','PE_name']
for i in newCol:
    df_gp.loc[:, i] = None
    
DMreg1 = ['PE_face','conflict','faceRep','nameRep','fullRepAlt']
DMreg2 = ['PE_name','conflict','faceRep','nameRep','fullRepAlt']
gp_result = pd.DataFrame()
df_gp['PE_face'] = abs(df_gp.conflict-df_gp.CP_face)
df_gp['PE_face'] = abs(df_gp.conflict-df_gp.CP_face)

#%%
for S in sbjList:
    df = df_gp[df_gp.sbjId==S]
    df.loc[df.faceId<=4, 'bkType'] = 1
    df.loc[df.faceId>=5, 'bkType'] = 2
    df.loc[:, 'runId'] = np.reshape(np.transpose(np.ones((int(len(df)/6),1),dtype=int)*np.arange(1,7,1)),[len(df),1])
    
    for r in np.unique(df.loc[:, 'runId']):
        T = df.loc[df.runId == r,:]
        for col in ['faceRep','nameRep']:
            idCol = 'faceId' if col=='faceRep' else 'nameId'
            T.loc[T.index[1:], col]=np.diff(np.array(T.loc[:,idCol]))
            T.loc[0,col]=1;
            T.loc[T[col]!=0,col]=1
            T.loc[:,col]=1-T.loc[:,col]
            T['fullRepAlt']=0
            T.loc[(T.faceRep==1) & (T.nameRep==1), 'fullRepAlt']=1
            T.loc[(T.faceRep==0) & (T.nameRep==0), 'fullRepAlt']=1    
        df.loc[df.runId== r, :]=T
      
    # end of all runs for a subject
    df['PE_face'] = abs(df.conflict-df.CP_face)
    df['PE_name'] = abs(df.conflict-df.CP_name)
    # get the weights for PE_face, PE_name for each subject

    sbjResult = pd.DataFrame()    
    sbjResult.loc[0,'sbjId'] = S
    for bkType in [1,2]:
        X1 = np.array(df.loc[df.bkType==bkType, DMreg1], dtype=float)
        X2 = np.array(df.loc[df.bkType==bkType, DMreg2], dtype=float)
        X1 = np.concatenate((X1, np.ones((len(X1),1))),axis=1)
        X2 = np.concatenate((X2, np.ones((len(X2),1))),axis=1)        
        
  #      RT = stats.zscore(df.loc[df.bkType==bkType, 'RT'])
        RT = df.loc[df.bkType==bkType, 'RT']
        gaussian_model1 = sm.GLM(RT, X1, family=sm.families.Gaussian())
        gaussian_results1 = gaussian_model1.fit()
        gaussian_model2 = sm.GLM(RT, X2, family=sm.families.Gaussian())
        gaussian_results2 = gaussian_model2.fit()
        if bkType ==1:
            sbjResult.loc[0,'SC_face']=gaussian_results1.tvalues[0]
            sbjResult.loc[0,'SC_name']=gaussian_results2.tvalues[0]
        else:
            sbjResult.loc[0,'SR_face']=gaussian_results1.tvalues[0]
            sbjResult.loc[0,'SR_name']=gaussian_results2.tvalues[0]
            
    gp_result = pd.concat((gp_result , sbjResult), axis=0)

# end of all subjects


#gp_result.to_pickle('gp_PEweights_v3.pkl')


