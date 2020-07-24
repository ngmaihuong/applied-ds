#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:32:31 2020

@author: QuynhMai
"""


import os #Change working directory
os.chdir('/Users/Boo Boo/Downloads/SUM20 Coursera/Applied Data Science with Python Specialization/Applied Plotting, Charting & Data Representation in Python/Data Files')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def housing_data():
    housing = pd.read_csv('ACSDP5Y2010.DP04_data_with_overlays_2020-07-23T141445.csv', skiprows=1, index_col='Geographic Area Name')
    housing = housing.drop('id', axis=1)
    housing = housing.T.drop('New York', axis=1)
    housing = housing.reset_index().rename(columns={'index': 'ind_vals', 'Queens County, New York': 'value'})
    housing[['vars','gr_vals','ind_vals']] = housing['ind_vals'].apply(lambda x: pd.Series(x.split('!!')))
    
    df = housing.set_index('gr_vals', 'ind_vals')
    df1 = df.where(df['vars'] == 'Estimate').dropna().drop('vars', axis=1)
    df2 = df.where(df['vars'] == 'Estimate Margin of Error').dropna().drop('vars', axis=1)
    df3 = df.where(df['vars'] == 'Percent').dropna().drop('vars', axis=1)
    df4 = df.where(df['vars'] == 'Percent Margin of Error').dropna().drop('vars', axis=1)
    
    df5 = pd.merge(df1, df2, on=['gr_vals', 'ind_vals'])
    df5 = df5.rename(columns={'value_x': 'estimate', 'value_y': 'estimateME'})
    
    df5 = pd.merge(df5, df3, on=['gr_vals', 'ind_vals'])
    df5 = pd.merge(df5, df4, on=['gr_vals', 'ind_vals'])
    df = df5.rename(columns={'value_x': 'percent', 'value_y': 'percentME'})
    
    df = df5.rename(columns={'value_x': 'estimate',
                             'value_y': 'estimateME',
                             'value_x': 'percent',
                             'value_y': 'percentME'})
    return df

#housing = housing_data()

def religion_data():
    relig = pd.read_excel('Longitudinal Religious Congregations and Membership File, 1980-2010 (County Level).XLSX')
    relig = relig.where(relig['STATEAB']=='NY').dropna().where(relig['CNTYNM']=='Queens County').dropna().drop(['FIPSMERG', 'CNTYNM', 'STATEAB'], axis=1)
    relig = relig.where(relig['YEAR'] >= 2000).dropna()
    relig['TOTPOP'] = relig['TOTPOP'].astype('int64')
    relig = relig.reset_index().drop(['NOTE_MIS', 'NOTE_COM', 'NOTE_MEA', 'index'], axis=1)
    
    return relig

religion = religion_data()

def demo_data():
    demo = pd.read_csv('ACSDP5Y2010.DP05_data_with_overlays_2020-07-23T161422.csv', skiprows=1, index_col='Geographic Area Name')
    demo = demo.drop('id', axis=1)
    demo = demo.T.drop('New York', axis=1)
    demo = demo.reset_index().rename(columns={'index': 'ind_vals', 'Queens County, New York': 'value'})
    demo[['vars','gr_vals','ind_vals', 'race', 'ethnicity']] = demo['ind_vals'].apply(lambda x: pd.Series(x.split('!!')))
    
    
    df = demo
    df1 = df.where(df['vars'] == 'Estimate').dropna(subset=['vars']).drop('vars', axis=1)
    df2 = df.where(df['vars'] == 'Estimate Margin of Error').dropna(subset=['vars']).drop('vars', axis=1)
    df3 = df.where(df['vars'] == 'Percent').dropna(subset=['vars']).drop('vars', axis=1)
    df4 = df.where(df['vars'] == 'Percent Margin of Error').dropna(subset=['vars']).drop('vars', axis=1)
       
    df5 = pd.merge(df1, df2, on=['gr_vals', 'ind_vals', 'race', 'ethnicity'])
    df5 = df5.rename(columns={'value_x': 'estimate', 'value_y': 'estimateME'})
    
    df5 = pd.merge(df5, df3, on=['gr_vals', 'ind_vals', 'race', 'ethnicity'])
    df5 = pd.merge(df5, df4, on=['gr_vals', 'ind_vals', 'race', 'ethnicity'])
    df = df5.rename(columns={'value_x': 'percent', 'value_y': 'percentME'})
    return df

#demog = demo_data()

#demog = demog.where(demog['gr_vals'] == 'RACE').dropna(subset=['gr_vals'])
#demog = demog.reset_index().drop('index', axis=1)
#demog = demog.drop([0, 1, 2, 3, 6, 11, 19])
#demog = demog.where(demog['ind_vals'] == 'One race').dropna(subset=['ind_vals']).drop(['gr_vals', 'ind_vals'], axis=1)

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

def census_data():
    census = pd.read_csv('co-est00int-alldata-36.csv')
    census = census.where(census['CTYNAME'] == 'Queens County').dropna()
    census = census[['YEAR', 'AGEGRP', 'TOT_POP', 'TOT_MALE', 'TOT_FEMALE', 'AA_MALE', 'AA_FEMALE', 'NHAA_MALE', 'NHAA_FEMALE', 'HAA_MALE', 'HAA_FEMALE']]
    census = census.replace({'YEAR': {1: 2000,
                                      2: 2000,
                                      3: 2001,
                                      4: 2002,
                                      5: 2003,
                                      6: 2004,
                                      7: 2005,
                                      8: 2006,
                                      9: 2007,
                                      10: 2008,
                                      11: 2009,
                                      12: 2010,
                                      13: 2010}})
    census = census[census.AGEGRP == 99]
    census = census.drop([10439, 10639])
    census = census.astype('int64')
    return census

census = census_data()

def dot_plot_adh():
    religion1 = religion.where(religion['YEAR'] == 2010).dropna()
    religion1['logADH'] = np.log(religion1['ADHERENT'])
    religion1 = religion1.sort_values(by='ADHERENT', ascending=False)
    #religion.plot('ADHERENT', 'CONGREG', kind='scatter')
    
    figure = plt.plot(religion1['logADH'], religion1['GRPNAME'], 'o')
    fig = plt.gcf()
    plt.grid()
    fig.set_size_inches(18, 25)
    return figure

religion1 = religion.where(religion['YEAR'] == 2010).dropna()
religion1['logADH'] = np.log(religion1['ADHERENT'])
religion1 = religion1.sort_values(by='ADHERENT', ascending=False)

labels = list(census['YEAR'])
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, census['AA_MALE'], width, label='Men', color='teal')
rects2 = ax.bar(x + width/2, census['AA_FEMALE'], width, label='Women', color='pink')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Estimated Population')
ax.set_title('Only Asian Population in Queens, New York \nby Gender from 2000 to 2010', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.tick_params(axis='x', length = 0)
ax.legend()

fig.tight_layout()

plt.show()