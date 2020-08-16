#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:20:26 2020

@author: christophenoblanc
"""

import os.path
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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

    # Add population rate
    df_grouped['rate_1M_pop']=df_grouped['value']/(df_grouped['total_pop']/1000000)
    df_grouped['rate_1M_pop_65up']=df_grouped['value']/(df_grouped['pop_65up']/1000000)

    # Last 5 day of the value
    last_days=5
    df_grouped['rolling_mean_delta'] = df_grouped.groupby('alpha-3')['delta_value'].rolling(last_days, min_periods=1).mean().reset_index(0,drop=True)
    df_grouped['rolling_mean_delta_1M_pop'] = df_grouped['rolling_mean_delta']/(df_grouped['total_pop']/1000000)

    # Alingn on the 1st day with 10 cases
    first_df=(df_grouped[df_grouped.value >= 10]).groupby(['alpha-3'])['date'].min().reset_index()
    first_df=first_df.rename(columns = {'date':'first_date'})
    df_grouped=pd.merge(df_grouped, first_df, on='alpha-3', how='left')
    
    #group_byday_df=df_grouped.dropna(subset=['first_date'])
    #group_byday_df['day'] = (group_byday_df['date'] - group_byday_df['first_date']).dt.days
    #group_byday_df=group_byday_df[group_byday_df.day >=0]
    group_byday_df=df_grouped
    group_byday_df['day'] = group_byday_df['date']
    
        
    return group_byday_df

# Prepare Daily stats by countries
countries_df=read_countrycode()
population_df=read_population("country_population.csv")

pop_total=(population_df[(population_df.series=='SP.POP.0014.TO') | (population_df.series=='SP.POP.1564.TO') \
                         | (population_df.series=='SP.POP.65UP.TO') ]).groupby(['alpha-3'])['value'].sum().reset_index()

confirmed_df=read_covid_19("time_series_covid19_confirmed_global.csv",countries_df,population_df)
death_df=read_covid_19("time_series_covid19_deaths_global.csv",countries_df,population_df)

# Filter One Country
filtered_Country=1
if filtered_Country == 1:
    confirmed_df=confirmed_df[ confirmed_df['Country']=="France" ]
    death_df=death_df[ death_df['Country']=="France" ]

lastday_refresh=death_df['date'].max()

# Sort & Country list
death_df.sort_values(by=['Country','day'], ascending=True, inplace=True)
confirmed_df.sort_values(by=['Country','day'], ascending=True, inplace=True)
#country_order=death_df.groupby(['Country'])['rate_1M_pop'].max().reset_index().sort_values(by='rate_1M_pop', ascending=False)

def showGraph(title,size=30,top=25,data=death_df,measure="value"):
    top_country=top
    order=data.groupby(['Country'])[measure].max().reset_index().sort_values(by=measure, ascending=False)
    order=order[:top_country]
    f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(15, 10))
    for a in order['Country']:
        df=data[data.Country == a][1:]
        plt.plot(df["day"], df[measure], label=a,marker=".")
        y_label=df[measure].max()
        index_max=np.argmax(np.array(df[measure]))
        x_label=df['day'].iat[index_max]
        #plt.annotate(xy=[df['day'].max(),df[measure].iat[-1]], s=a)
        if filtered_Country !=1:
            plt.annotate(xy=[x_label,y_label], s=a)
        if filtered_Country==1:
            plt.annotate(xy=[df['day'].iat[-1],df[measure].iat[-1]], s=df[measure].iat[-1])
        
    plt.annotate(xy=(0.5, 0.9), xycoords="axes fraction", s=title,size=size,ha='center', va='center')
    filtered_Country
    #labelLines(plt.gca().get_lines(), zorder=2.5)
    #labelLines(plt.gca().get_lines(), align=False, color='k')
    #plt.xticks(rotation=70)
    f.autofmt_xdate(bottom=0.2, rotation=45, ha='right')
    ax0.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0.xaxis.set_minor_locator(mdates.DayLocator())
    ax0.set_ylabel(title)
    ax0.set_xlabel('date')
    ax0.set_title('%s (top %i countries, refreshed %s)'%(title,top_country,lastday_refresh.strftime('%d %b %Y')))
    plt.legend(loc="upper left")
    plt.show()
        
# -----------------------
# ---- # of DEATH -------
# -----------------------
# Show Number of Death by Day count since 10th Death
showGraph(data=death_df,measure="value",title="# of DEATH",size=30,top=30)

# By  1 Million population
showGraph(data=death_df,measure="rate_1M_pop",title="# of DEATH by 1 million population",size=30,top=30)

# --------------------------------------------
# ---- DEATH rate (mean of last 5 days) -------
# --------------------------------------------
# DEATH cases : Show Number of Mean of Last 5 days of Delta Death by Date
showGraph(data=death_df,measure="rolling_mean_delta",title="Rolling mean of last 5 days daily DEATH",size=22,top=30)

# By 1 million population
showGraph(data=death_df,measure="rolling_mean_delta_1M_pop",title="Rolling mean of last 5 days daily DEATH (per 1M)",size=20,top=30)

# ---------------------------
# ---- # of CONFIRMED -------
# ---------------------------
# Show Number of Confirmed by Date
showGraph(data=confirmed_df,measure="value",title="# of CONFIRMED",size=30,top=30)

# By  1 Million population
showGraph(data=confirmed_df,measure="rate_1M_pop",title="# of CONFIRMED by 1M population",size=30,top=30)

# -------------------------------------------------
# ---- CONFIRMED rate (mean of last 5 days) -------
# -------------------------------------------------
# CONFIRMED cases : Show Number of Mean of Last 5 days of Delta Death by Date
showGraph(data=confirmed_df,measure="rolling_mean_delta",title="Rolling mean of last 5 days daily CONFIRMED",size=22,top=30)

# By 1 million population
showGraph(data=confirmed_df,measure="rolling_mean_delta_1M_pop",title="Rolling mean of last 5 days daily CONFIRMED (per 1M)",size=18,top=30)

# -------------------------------------------------
# ---- Loop on All the Countries -------
# -------------------------------------------------
count_c=0
for c in confirmed_df.Country.unique():
    count_c=count_c+1
    if count_c>-1 and count_c<=900:
        confirmed_df_c=confirmed_df[ confirmed_df['Country']==c ]
        death_df_c=death_df[ death_df['Country']==c ]
        showGraph(data=death_df_c,measure="rolling_mean_delta",title="Rolling mean of last 5 days daily DEATH for "+c,size=30,top=30)
        #showGraph(data=confirmed_df_c,measure="rolling_mean_delta",title="Rolling mean of last 5 days daily CONFIRMED for "+c,size=30,top=30)


