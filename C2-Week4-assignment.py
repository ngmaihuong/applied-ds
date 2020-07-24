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

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

def religion_data():
    relig = pd.read_excel('Longitudinal Religious Congregations and Membership File, 1980-2010 (County Level).XLSX')
    relig = relig.where(relig['STATEAB']=='NY').dropna().where(relig['CNTYNM']=='Queens County').dropna().drop(['FIPSMERG', 'CNTYNM', 'STATEAB'], axis=1)
    relig = relig.where(relig['YEAR'] >= 2000).dropna()
    relig['TOTPOP'] = relig['TOTPOP'].astype('int64')
    relig = relig.reset_index().drop(['NOTE_MIS', 'NOTE_COM', 'NOTE_MEA', 'index'], axis=1)
    
    return relig

religion = religion_data()

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

def religion_lollipop():
    religion1 = religion.where(religion['YEAR'] == 2010).dropna()
    religion1['logADH'] = np.log(religion1['ADHERENT'])
    religion1 = religion1.sort_values(by='ADHERENT', ascending=False)
    #religion.plot('ADHERENT', 'CONGREG', kind='scatter')
    
    import seaborn as sns
    plt.hlines(y=religion1['GRPNAME'], xmin=0, xmax=religion1['logADH'], color='skyblue')
    figure = plt.plot(religion1['logADH'], religion1['GRPNAME'], 'o')
    fig = plt.gcf()
    #plt.grid()
    fig.set_size_inches(18, 25)
    return figure

def aa_pop_bar():
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
    return fig 
    
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

