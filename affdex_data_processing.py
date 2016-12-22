# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version




"""
Begin data collection
"""


df_gb1 = pandas.read_csv('/Users/gautambhat/Downloads/video_gb1_labelled.csv')
df_gb4 = pandas.read_csv('/Users/gautambhat/Downloads/video_gb4_labelled.csv')
df_gb6 = pandas.read_csv('/Users/gautambhat/Downloads/video_gb6_labelled.csv')
df_gb7 = pandas.read_csv('/Users/gautambhat/Downloads/video_gb7_labelled.csv')
#dataset = np.loadtxt('/Users/gautambhat/Downloads/video_gb4_labelled.csv', delimiter=",")


"""
Change mapping of yes and no to 1 and 0 in glasses column
"""
yes_no_map = { 'yes' : 1, 'no' : 0}

df_gb1['glasses'] = df_gb1['glasses'].map(yes_no_map)
df_gb4['glasses'] = df_gb4['glasses'].map(yes_no_map)
df_gb6['glasses'] = df_gb6['glasses'].map(yes_no_map)
df_gb7['glasses'] = df_gb7['glasses'].map(yes_no_map)




"""
Drop irrelevant features from sample data
"""
df_gb1.drop(df_gb1[['faceId','age','ethnicity', 'gender', 'dominantEmoji']],axis=1, inplace=True)
df_gb4.drop(df_gb4[['faceId','age','ethnicity', 'gender', 'dominantEmoji']],axis=1, inplace=True)
df_gb6.drop(df_gb6[['faceId','age','ethnicity', 'gender', 'dominantEmoji']],axis=1, inplace=True)
df_gb7.drop(df_gb7[['faceId','age','ethnicity', 'gender', 'dominantEmoji']],axis=1, inplace=True)




"""
Reformat Timestamp to remove nan values
"""
df_gb1['TimeStamp'] = df_gb1['TimeStamp'].astype(str).str.replace('00nan','')
df_gb4['TimeStamp'] = df_gb4['TimeStamp'].astype(str).str.replace('00nan','')
df_gb6['TimeStamp'] = df_gb6['TimeStamp'].astype(str).str.replace('00nan','')
df_gb7['TimeStamp'] = df_gb7['TimeStamp'].astype(str).str.replace('00nan','')





"""
Convert string objects to float64 for numeric use
"""
df_gb1['TimeStamp'] = pandas.to_numeric(df_gb1['TimeStamp'])
df_gb4['TimeStamp'] = pandas.to_numeric(df_gb4['TimeStamp'])
df_gb6['TimeStamp'] = pandas.to_numeric(df_gb6['TimeStamp'])
df_gb7['TimeStamp'] = pandas.to_numeric(df_gb7['TimeStamp'])




"""
Drop first two seconds of frames, assuming they won't be exactly significant
"""
df_gb1 = df_gb1[df_gb1['TimeStamp'] >= 2]
df_gb4 = df_gb4[df_gb4['TimeStamp'] >= 2]
df_gb6 = df_gb6[df_gb6['TimeStamp'] >= 2]
df_gb7 = df_gb7[df_gb7['TimeStamp'] >= 2]




"""
Reset indices
"""
df_gb1.reset_index(drop=True, inplace=True)
df_gb4.reset_index(drop=True, inplace=True)
df_gb6.reset_index(drop=True, inplace=True)
df_gb7.reset_index(drop=True, inplace=True)




"""
Create target columns as separate dataframe for later use
"""
target_gb1 = df_gb1[['TimeStamp', 'FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']]
target_gb4 = df_gb4[['TimeStamp', 'FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']]
target_gb6 = df_gb6[['TimeStamp', 'FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']]
target_gb7 = df_gb7[['TimeStamp', 'FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']]




"""
Drop target features
"""
df_gb1.drop(df_gb1[['FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']],axis=1, inplace=True)
df_gb4.drop(df_gb4[['FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']],axis=1, inplace=True)
df_gb6.drop(df_gb6[['FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']],axis=1, inplace=True)
df_gb7.drop(df_gb7[['FRUSTRATED', 'TIRED', 'ENGAGED', 'CLASS']],axis=1, inplace=True)




#"""
#Drop Nan
#"""
#df_gb1 = df_gb1.dropna(subset=['glasses'])
#df_gb4 = df_gb4.dropna(subset=['glasses'])
#df_gb6 = df_gb6.dropna(subset=['glasses'])
#df_gb7 = df_gb7.dropna(subset=['glasses'])
#
#
#
#
#"""
#Converting interocular distance to float, now that nan has been removed
#"""
#df_gb1['interocularDistance'] = pandas.to_numeric(df_gb1['interocularDistance'])
#df_gb4['interocularDistance'] = pandas.to_numeric(df_gb4['interocularDistance'])
#df_gb6['interocularDistance'] = pandas.to_numeric(df_gb6['interocularDistance'])
#df_gb7['interocularDistance'] = pandas.to_numeric(df_gb7['interocularDistance'])




"""
Aggregate rows every 0.3 seconds
"""

#gb1
xInter_gb1 = pandas.DataFrame()
yInter_gb1 = pandas.DataFrame()
currIndex= 0
while currIndex < target_gb1.index.size :
    start_time = target_gb1.loc[currIndex,'TimeStamp']
    current_frames = target_gb1[target_gb1['TimeStamp'].between(start_time,start_time+0.3, inclusive=True)]
    xAgg = df_gb1.ix[current_frames.index]
    xAgg = xAgg.dropna(subset=['glasses'])
    xAgg['interocularDistance'] = pandas.to_numeric(xAgg['interocularDistance'])
    xAgg = xAgg.mean()
    yAgg = current_frames.mode()
    xAgg['TimeStamp'] = start_time
    yAgg['TimeStamp'] = start_time
    print(yAgg['TimeStamp'],'\n')
    
    xInter_gb1 = xInter_gb1.append(xAgg,ignore_index=True);
    yInter_gb1 = yInter_gb1.append(yAgg, ignore_index=True);

    currIndex = current_frames.index[current_frames.index.size - 1] + 1

result_gb1 = xInter_gb1.merge(yInter_gb1, left_on='TimeStamp', right_on='TimeStamp', how='outer')
result_gb1 = result_gb1.dropna(subset=['anger'])


#gb4
mms = MinMaxScaler()
xInter_gb4 = pandas.DataFrame()
yInter_gb4 = pandas.DataFrame()
currIndex= 0
while currIndex < target_gb4.index.size :
    start_time = target_gb4.loc[currIndex,'TimeStamp']
    current_frames = target_gb4[target_gb4['TimeStamp'].between(start_time,start_time+0.4, inclusive=True)]
    xAgg = df_gb4.ix[current_frames.index]
    xAgg = xAgg.dropna(subset=['glasses'])
    xAgg['interocularDistance'] = pandas.to_numeric(xAgg['interocularDistance'])
    xAgg = xAgg.mean()
    yAgg = current_frames.mode()
    xAgg['TimeStamp'] = start_time
    yAgg['TimeStamp'] = start_time
    print(yAgg['TimeStamp'],'\n')
    
    xInter_gb4 = xInter_gb4.append(xAgg,ignore_index=True);
    yInter_gb4 = yInter_gb4.append(yAgg, ignore_index=True);

    currIndex = current_frames.index[current_frames.index.size - 1] + 1

result_gb4 = xInter_gb4.merge(yInter_gb4, left_on='TimeStamp', right_on='TimeStamp', how='outer')
result_gb4 = result_gb4.dropna(subset=['anger'])


#gb6
xInter_gb6 = pandas.DataFrame()
yInter_gb6 = pandas.DataFrame()
currIndex= 0
while currIndex < target_gb6.index.size :
    start_time = target_gb6.loc[currIndex,'TimeStamp']
    current_frames = target_gb6[target_gb6['TimeStamp'].between(start_time,start_time+0.4, inclusive=True)]
    xAgg = df_gb6.ix[current_frames.index]
    xAgg = xAgg.dropna(subset=['glasses'])
    xAgg['interocularDistance'] = pandas.to_numeric(xAgg['interocularDistance'])
    xAgg = xAgg.mean()
    yAgg = current_frames.mode()
    xAgg['TimeStamp'] = start_time
    yAgg['TimeStamp'] = start_time
    print(yAgg['TimeStamp'],'\n')
    
    xInter_gb6 = xInter_gb6.append(xAgg,ignore_index=True);
    yInter_gb6 = yInter_gb6.append(yAgg, ignore_index=True);

    currIndex = current_frames.index[current_frames.index.size - 1] + 1

result_gb6 = xInter_gb6.merge(yInter_gb6, left_on='TimeStamp', right_on='TimeStamp', how='outer')
result_gb6 = result_gb6.dropna(subset=['anger'])


#gb7
xInter_gb7 = pandas.DataFrame()
yInter_gb7 = pandas.DataFrame()
currIndex= 0
while currIndex < target_gb7.index.size :
    start_time = target_gb7.loc[currIndex,'TimeStamp']
    current_frames = target_gb7[target_gb7['TimeStamp'].between(start_time,start_time+0.4, inclusive=True)]
    xAgg = df_gb7.ix[current_frames.index]
    xAgg = xAgg.dropna(subset=['glasses'])
    xAgg['interocularDistance'] = pandas.to_numeric(xAgg['interocularDistance'])
    xAgg = xAgg.mean()
    yAgg = current_frames.mode()
    xAgg['TimeStamp'] = start_time
    yAgg['TimeStamp'] = start_time
    print(yAgg['TimeStamp'],'\n')
    
    xInter_gb7 = xInter_gb7.append(xAgg,ignore_index=True);
    yInter_gb7 = yInter_gb7.append(yAgg, ignore_index=True);

    currIndex = current_frames.index[current_frames.index.size - 1] + 1

result_gb7 = xInter_gb7.merge(yInter_gb7, left_on='TimeStamp', right_on='TimeStamp', how='outer')
result_gb7 = result_gb7.dropna(subset=['anger'])



"""
Create new feature set with n-gram data
"""
window_size = 6

R_gb1 = pandas.concat([result_gb1.iloc[:,0].shift(-window_size + 1)] + [result_gb1.iloc[:,1:47].shift(-i) for i in range(window_size)] + [result_gb1.iloc[:,48:52].shift(-window_size + 1)], axis=1).iloc[0:-window_size + 1]
R_gb4 = pandas.concat([result_gb4.iloc[:,0].shift(-window_size + 1)] + [result_gb4.iloc[:,1:47].shift(-i) for i in range(window_size)] + [result_gb4.iloc[:,48:52].shift(-window_size + 1)], axis=1).iloc[0:-window_size + 1]
R_gb6 = pandas.concat([result_gb6.iloc[:,0].shift(-window_size + 1)] + [result_gb6.iloc[:,1:47].shift(-i) for i in range(window_size)] + [result_gb6.iloc[:,48:52].shift(-window_size + 1)], axis=1).iloc[0:-window_size + 1]
R_gb7 = pandas.concat([result_gb7.iloc[:,0].shift(-window_size + 1)] + [result_gb7.iloc[:,1:47].shift(-i) for i in range(window_size)] + [result_gb7.iloc[:,48:52].shift(-window_size + 1)], axis=1).iloc[0:-window_size + 1]

result = R_gb1.append(R_gb4, ignore_index=True)
result = result.append(R_gb6, ignore_index=True)
result = result.append(R_gb7, ignore_index=True)


print(result.shape,'\n')

result.to_csv('/Users/gautambhat/Downloads/finalResult.csv')


