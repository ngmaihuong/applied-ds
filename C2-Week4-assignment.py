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
    relig = relig.reset_index().drop(['RELTRAD', 'FAMILY', 'NOTE_MIS', 'NOTE_COM', 'NOTE_MEA', 'index'], axis=1)
    relig = relig.reset_index().drop('index', axis=1)
    relig['TOTPOP'] = relig['TOTPOP'].astype('int64')
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
    census = census.reset_index().drop('index', axis=1)
    census['TOT_AA'] =  census[['AA_MALE', 'AA_FEMALE', 'NHAA_MALE', 'NHAA_FEMALE', 'HAA_MALE', 'HAA_FEMALE']].sum(axis=1)
    census['AA_PER'] = census['TOT_AA'] / census['TOT_POP']
    return census

census = census_data()

def religion_race_data():
    religrace = pd.read_excel('AnnualRaceEthnicityAdherents.xlsx',
                              usecols = [0, 1, 2, 9],
                              sheet_name='2010', 
                              skipfooter=5)
    religrace = religrace[religrace['Asian/Pacific Islanders'] != 0]
    religrace = religrace.reset_index().drop('index', axis=1)
    religrace = religrace.replace({'Name': 
                                   {'Int Pentecostal Holiness Church': 'International Pentecostal Holiness Church'}})    
    return religrace

rare = religion_race_data()

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

#relig1 = religion.sort_values(by='GRPNAME', ascending=True)
#rare1 = rare.sort_values(by='Name', ascending=True)
#relig1 = religion.where(religion['GRPNAME'].str.contains(rare['Name'].iloc[0])).dropna()
#relig1 = relig1.append(religion.where(religion['GRPNAME'].str.contains(rare['Name'].iloc[2][4:12])).dropna())

def aa_relig_match():
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    
    def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=2):
        s = df_2[key2].tolist()
    
        m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
        df_1['matches'] = m
    
        m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
        df_1['matches'] = m2
    
        return df_1
    
    rare_match = fuzzy_merge(religion, rare, 'GRPNAME', 'Name', threshold=90)
    rare_match = rare_match.where(rare_match['matches'] != '').dropna(subset=['matches'])
    rare_match = pd.merge(rare_match, rare, left_on='matches', right_on='Name')
    
    rare_match = rare_match.where(rare_match['YEAR'] >= 2000).dropna(subset=['YEAR'])
    rare_match = rare_match.drop(list(rare_match[rare_match['GRPCODE'].str.contains('[A-Za-z]')].index))
    rare_match['Name'] = rare_match['Name'].str.replace(r"\s*\(.*\)\s*.*", "")
    rare_match = rare_match.drop(['GRPNAME', 'matches'], axis=1)
    rare_match = rare_match.reset_index().drop('index', axis=1)
    rare_match['aa_ad_p'] = rare_match['Asian/Pacific Islanders'] / rare_match['Total Adherents']
    rare_match['queens_ad_p'] = rare_match['ADHERENT'] / rare_match['TOTPOP']
    rare_match = rare_match.where(rare_match['queens_ad_p'] >= 0.01).dropna(subset=['queens_ad_p'])
    rare_match['TOTPOP'] = rare_match['TOTPOP'].astype('int64')
    rare_match['Asian/Pacific Islanders'] = rare_match['Asian/Pacific Islanders'].astype('int64')
    rare_match['Total Adherents'] = rare_match['Total Adherents'].astype('int64')
    return rare_match

rarelig = aa_relig_match()

labels = list(rarelig['YEAR'].astype('int64'))

p_2000 = ( census['AA_PER'].iloc[0] * rarelig['queens_ad_p'].iloc[0] )
p_2010 = ( census['AA_PER'].iloc[10] * rarelig['queens_ad_p'].iloc[1] )

rarelig['p_queens_aa_ad'] = [p_2000, p_2010]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis

ax1.bar(census['YEAR'], census['TOT_AA'], color='teal', alpha=0.5, label='Asian Population in Queens, NY')

ax2.plot(census['YEAR'],census['AA_PER'], '-o', color='gold', scaley=False, label='Percentage of Asian Population in Queens, NY')
ax2.plot(rarelig['YEAR'], rarelig['queens_ad_p'], '-o', color='navy', scaley=False, label='Percentage of Catholic Adherents in Queens, NY')
ax2.plot(rarelig['YEAR'], rarelig['p_queens_aa_ad'], '-o', color='maroon', scaley=False, label = 'Probability of one being both Asian and Catholic in Queens, NY')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), frameon=False)

plt.title('Asian Catholic Population in Queens, NY from 2000 to 2010', fontweight='bold')

fig.set_size_inches(8, 5)
plt.show()

