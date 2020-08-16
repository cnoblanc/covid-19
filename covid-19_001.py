#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:20:26 2020

@author: christophenoblanc
"""

import os.path
import pandas as pd
from datetime import datetime,timedelta

def read_countrycode():
    url = 'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/'
    infile=url+'all.csv'
    df=pd.read_csv(infile, parse_dates=[4],skip_blank_lines=False )
    df=df.rename(columns = {'name':'Country'})
    
    # Update some Country names  
    df.Country[df.Country == 'Bolivia (Plurinational State of)'] = 'Bolivia'
    df.Country[df.Country == 'Brunei Darussalam'] = 'Brunei'
    df.Country[df.Country == 'Congo, Democratic Republic of the'] = 'Congo'
    df.Country[df.Country == 'Iran (Islamic Republic of)'] = 'Iran'
    
    df.Country[df.Country == 'Korea, Republic of'] = 'Korea, South'
    df.Country[df.Country == 'Moldova, Republic of'] = 'Moldova'
    df.Country[df.Country == 'Russian Federation'] = 'Russia'
    df.Country[df.Country == 'Taiwan, Province of China'] = 'Taiwan'
    df.Country[df.Country == 'Tanzania, United Republic of'] = 'Tanzania'
    df.Country[df.Country == 'United Kingdom of Great Britain and Northern Ireland'] = 'United Kingdom'
    df.Country[df.Country == 'United States of America'] = 'USA'
    df.Country[df.Country == 'Venezuela (Bolivarian Republic of)'] = 'Venezuela'
    df.Country[df.Country == 'Syrian Arab Republic'] = 'Syria'
    df.Country[df.Country == "Lao People's Democratic Republic"] = 'Laos'
    #df.Country[df.Country == ''] = ''
    return df


def read_population(csv_file):
    #base_path="data/"
    base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/covid-19/data/"
    infile=base_path+csv_file
    df=pd.read_csv(infile, parse_dates=[4],skip_blank_lines=False )
    df=df.drop(columns=['2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2019 [YR2019]'])
    df=df.rename(columns = {'Country Code':'alpha-3', '2018 [YR2018]':'value', 'Series Code':'series'})
    df.dropna(subset=['alpha-3'],inplace=True)
    df=df[df.value !='..']
    df.value = pd.to_numeric(df.value, errors='coerce')
    
    #pop_total=(df[(df.series=='SP.POP.0014.TO') | (df.series=='SP.POP.1564.TO')| (df.series=='SP.POP.65UP.TO') ]).groupby(['alpha-3'])['value'].sum().reset_index()
    #pop_total=pop_total.rename(columns = {'value':'total_pop'})
    return df


def read_covid_19(csv_file,countries_df,population_df):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
    infile=url+csv_file
    df=pd.read_csv(infile, parse_dates=[4],skip_blank_lines=False )
    
    # Prepare the Country column
    df=df.rename(columns = {'Country/Region':'Country'})
    df.Country[df.Country == "Cote d'Ivoire"] = "Côte d'Ivoire"
    df.Country[df.Country == 'Taiwan*'] = 'Taiwan'
    df.Country[df.Country == 'US'] = 'USA'
    df.Country[df.Country == 'Vietnam'] = 'Viet Nam'
    #df.Country[df.Country == ''] = ''
    # List of Country excluded (need to make something ?):
    # Congo (Brazzaville), Congo (Kinshasa), West Bank and Gaza
    # Diamond Princess
    
    # Merge covid-19 with country codes
    df=pd.merge(df, countries_df[['Country','alpha-3','region','sub-region']], on='Country', how='left')
    df=df.rename(columns = {'alpha-3_x':'alpha-3', 'region_x':'region','sub-region_x':'sub-region'})
    df.dropna(subset=['alpha-3'],inplace=True)

    # Merge with population
    pop_total=(population_df[(population_df.series=='SP.POP.0014.TO') | (population_df.series=='SP.POP.1564.TO') \
                             | (population_df.series=='SP.POP.65UP.TO') ]).groupby(['alpha-3'])['value'].sum().reset_index()
    pop_total=pop_total.rename(columns = {'value':'total_pop'})
    df=pd.merge(df, pop_total, on='alpha-3', how='left')
    
    # Merge with age +65
    pop_65up=(population_df[(population_df.series=='SP.POP.65UP.TO') ]).groupby(['alpha-3'])['value'].sum().reset_index()
    pop_65up=pop_65up.rename(columns = {'value':'pop_65up'})
    df=pd.merge(df, pop_65up, on='alpha-3', how='left')

    # Transpose the date columns into rows
    id_cols=['Province/State', 'Country', 'Lat', 'Long','alpha-3','region','sub-region','total_pop','pop_65up']
    df=df.melt(id_vars=id_cols)
    df['date']=pd.to_datetime(df['variable'])
    df=df.drop(columns='variable')
    df.value = pd.to_numeric(df.value, errors='coerce')

    # Group by Province/State
    df_grouped=df.groupby(['Country','alpha-3','region','sub-region','total_pop','pop_65up','date'])['value'].sum().reset_index()

    # Merge with Last day value
    df_grouped['date_yesterday']=df_grouped['date'] - pd.to_timedelta(1, unit='d')
    yesterday_df=df_grouped[['alpha-3','date','value']].copy()
    yesterday_df=yesterday_df.rename(columns = {'date':'date_yesterday', 'value':'yesterday_value'})
    df_grouped=pd.merge(df_grouped, yesterday_df, on=['alpha-3','date_yesterday'], how='left')
    df_grouped=df_grouped.drop(columns=['date_yesterday'])
    df_grouped['delta_value']=df_grouped['value']-df_grouped['yesterday_value']

    last_days=5
    df_grouped['rolling_mean_delta'] = df_grouped.groupby('alpha-3')['delta_value'].rolling(last_days, min_periods=1).mean().reset_index(0,drop=True)

    first_df=(df_grouped[df_grouped.value >= 10]).groupby(['alpha-3'])['date'].min().reset_index()
    first_df=first_df.rename(columns = {'date':'first_date'})
    df_grouped=pd.merge(df_grouped, first_df, on='alpha-3', how='left')
    
    group_byday_df=death_df.dropna(subset=['first_date'])
    group_byday_df['day'] = (group_byday_df['date'] - group_byday_df['first_date']).dt.days
    group_byday_df=group_byday_df[group_byday_df.day >=0]
    
    # Add population rate
    group_byday_df['rate_1M_pop']=group_byday_df['value']/(group_byday_df['total_pop']/1000000)
    group_byday_df['rate_1M_pop_65up']=group_byday_df['value']/(group_byday_df['pop_65up']/1000000)

    return df_grouped

# Prepare Daily stats by countries
countries_df=read_countrycode()
population_df=read_population("country_population.csv")

pop_total=(population_df[(population_df.series=='SP.POP.0014.TO') | (population_df.series=='SP.POP.1564.TO') \
                         | (population_df.series=='SP.POP.65UP.TO') ]).groupby(['alpha-3'])['value'].sum().reset_index()

confirmed_df=read_covid_19("time_series_covid19_confirmed_global.csv",countries_df,population_df)
death_df=read_covid_19("time_series_covid19_deaths_global.csv",countries_df,population_df)
lastday_refresh=death_df['date'].max()

# Get stats for the Last x days
df_grouped=confirmed_df.copy()
#df_grouped.reset_index(inplace=True)    # This gives us index 0,1,2... and a new col 'index'
#df_grouped.set_index('date', inplace=True)    # Replace with date in yours
# This next bit does the groupby and rolling, which will give a df 
# with a multi index of foo and baz, then reset_index(0) to remove the foo index level
# so that it matches the original df index so that you can add it as a new column
df_grouped['rolling'] = df_grouped.groupby('alpha-3')['value'].rolling(5, min_periods=1).mean().reset_index(0,drop=True)
#df_grouped.reset_index(inplace=True)    # brings baz back into the df as a column
#df.set_index('index', inplace=True)   # sets the index back to the original



# Get the offset by the first day of 10 death
firstDeath_df=(death_df[death_df.value >= 10]).groupby(['alpha-3'])['date'].min().reset_index()
firstDeath_df=firstDeath_df.rename(columns = {'date':'first_death_date'})
death_df=pd.merge(death_df, firstDeath_df, on='alpha-3', how='left')

death_byday_df=death_df.dropna(subset=['first_death_date'])
death_byday_df['day'] = (death_byday_df['date'] - death_byday_df['first_death_date']).dt.days
death_byday_df=death_byday_df[death_byday_df.day >=0]

# Add population rate
death_byday_df['rate_1M_pop']=death_byday_df['value']/(death_byday_df['total_pop']/1000000)
death_byday_df['rate_1M_pop_65up']=death_byday_df['value']/(death_byday_df['pop_65up']/1000000)


# Sort & Country list
death_byday_df.sort_values(by=['Country','day'], ascending=True, inplace=True)
country_order=death_byday_df.groupby(['Country'])['value'].max().reset_index().sort_values(by='value', ascending=False)


# Show Number of Death by Day count since 10th Death
import seaborn as sns
import matplotlib.pyplot as plt
country_order=death_byday_df.groupby(['Country'])['value'].max().reset_index().sort_values(by='value', ascending=False)
top_country=25
country_order=country_order[:top_country]
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
for a in country_order['Country']:
    data=death_byday_df[death_byday_df.Country == a]
    plt.plot(data["day"], data["value"], label=a,marker=".")
    plt.annotate(xy=[data['day'].max(),data['value'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylabel('death number')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()

top_country=15
country_order=country_order[:top_country]
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
cut_death_to=1000
for a in country_order['Country']:
    data=death_byday_df[ (death_byday_df.Country == a)  ]
    plt.plot(data["day"], data["value"], label=a,marker=".")
    plt.annotate(xy=[data['day'].max(),data['value'].max()], s=a)
    plt.annotate(xy=[data[data["value"]<cut_death_to]['day'].max(),data[data["value"]<cut_death_to]['value'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylim([0, cut_death_to])
ax0.set_xlim([0, 31])
ax0.set_ylabel('death number')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death since 10th death (top %i countries)'%(top_country))
plt.legend(loc="upper left")
plt.show()


# By  rate (population rate)
#country_order=death_byday_df.groupby(['Country'])['rate_1M_pop'].max().reset_index().sort_values(by='rate_1M_pop', ascending=False)
top_country=25
country_order=country_order[:top_country]
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
for a in country_order['Country']:
    data=death_byday_df[death_byday_df.Country == a]
    plt.plot(data["day"], data["rate_1M_pop"], label=a,marker=".")
    plt.annotate(xy=[data['day'].max(),data['rate_1M_pop'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylabel('death count by 1 million population')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death by 1 million population since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()

f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
cut_death_to=30
for a in country_order['Country']:
    data=death_byday_df[ (death_byday_df.Country == a) ]
    plt.plot(data["day"], data["rate_1M_pop"], label=a,marker=".")
    plt.annotate(xy=[data[data["rate_1M_pop"]<cut_death_to]['day'].max(),data[data["rate_1M_pop"]<cut_death_to]['rate_1M_pop'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylim([0, cut_death_to])
#ax0.set_xlim([0, 31])
ax0.set_ylabel('death count by 1 million population')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death by 1 million population since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()


# By  rate (population rate : age 65 and above)
#country_order=death_byday_df.groupby(['Country'])['rate_1M_pop_65up'].max().reset_index().sort_values(by='rate_1M_pop_65up', ascending=False)
top_country=15
country_order=country_order[:top_country]
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
for a in country_order['Country']:
    data=death_byday_df[death_byday_df.Country == a]
    plt.plot(data["day"], data["rate_1M_pop_65up"], label=a,marker=".")
    plt.annotate(xy=[data['day'].max(),data['rate_1M_pop_65up'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylabel('death count by "1 million population of aged 65+" ')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death by "1 million population aged 65+" since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()

f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
cut_death_to=150
for a in country_order['Country']:
    data=death_byday_df[ (death_byday_df.Country == a)  ]
    plt.plot(data["day"], data["rate_1M_pop_65up"], label=a,marker=".")
    plt.annotate(xy=[data[data["rate_1M_pop_65up"]<cut_death_to]['day'].max(),data[data["rate_1M_pop_65up"]<cut_death_to]['rate_1M_pop_65up'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylim([0, cut_death_to])
#ax0.set_xlim([0, 31])
ax0.set_ylabel('death count by "1 million population of aged 65+" ')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Number of death by "1 million population aged 65+" since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()


# Show Number of Mean of Last 5 days of Delta Death by Day count since 10th Death
import seaborn as sns
import matplotlib.pyplot as plt
country_order=death_byday_df.groupby(['Country'])['value'].max().reset_index().sort_values(by='value', ascending=False)
top_country=25
country_order=country_order[:top_country]
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
for a in country_order['Country']:
    data=death_byday_df[death_byday_df.Country == a]
    plt.plot(data["day"], data["rolling_mean_delta"], label=a,marker=".")
    plt.annotate(xy=[data['day'].max(),data['rolling_mean_delta'].max()], s=a)
#labelLines(plt.gca().get_lines(), zorder=2.5)
#labelLines(plt.gca().get_lines(), align=False, color='k')
ax0.set_ylabel('rolling_mean_delta = (mean of additional death from previous 5 days)')
ax0.set_xlabel('days since 10th death')
ax0.set_title('Mean of # of daily additional death (delta death) since 10th death (top %i countries, refreshed %s)'%(top_country,lastday_refresh.strftime('%d %b %Y')))
plt.legend(loc="upper left")
plt.show()


