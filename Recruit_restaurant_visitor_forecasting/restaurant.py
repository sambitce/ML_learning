# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:11:56 2018

@author: sambi
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import calendar
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry import Point
from shapely.wkt import loads
import plotly.plotly as py
from shapely.geometry import Polygon

def load_data(data_file_name):
    return pd.read_csv(data_file_name)

def create_train_test_data(data):
    return train_test_split(data,
                            test_size=0.2,
                            random_state=42)
    
air_visit_data = load_data('air_visit_data.csv')
print(air_visit_data.head())
print(air_visit_data.info())

print(air_visit_data["air_store_id"].value_counts())
print(air_visit_data["visit_date"].value_counts())
print(air_visit_data.describe())

air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
print("After")
print(air_visit_data.info())


air_visit_data['Month'] = air_visit_data['visit_date'].dt.month
print(air_visit_data['Month'])


print("end")

air_store_info = load_data('air_store_info.csv')

print(air_store_info.head())
print(air_store_info.describe())

#air_visit_data.hist(bins=50 , figsize=(100,15))
#plt.show()

air_visit_store_data = pd.merge(left=air_visit_data,
                                right=air_store_info,
                                left_on='air_store_id',
                                right_on='air_store_id' )

train_set , test_set = create_train_test_data(air_visit_store_data)

copy_air_visit_data_train = train_set.copy()
print(copy_air_visit_data_train.head())


'''
copy_air_visit_data_train.plot(kind="scatter" , x="latitude" , 
                              y="longitude",
                              alpha=0.4,
                              s=copy_air_visit_data_train["visitors"]*10,
                              label="Total visitors" ,
                              c="visitors" ,
                              cmap=plt.get_cmap("jet"),
                              colorbar=True,
                              figsize=(13,10))
'''
#plt.legend()
#plt.plot(copy_air_visit_data_train.visit_date,copy_air_visit_data_train.visitors)

#plt.plot(copy_air_visit_data_train.groupby('visit_date')['visitors'].sum(),  )
#fig, ax = plt.subplots(figsize=(15,7))
#copy_air_visit_data_train.groupby('visit_date')['visitors'].sum().plot(ax=ax)

group_by_genre = copy_air_visit_data_train.groupby(['visit_date','air_genre_name']).sum()['visitors'].unstack()

#group_by_genre.plot(kind='line', stacked=True, figsize=[15,6], colormap='gist_rainbow')




days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
months =['Jan' ,'Feb','Mar','Apr','May','Jun',
         'Jul','Aug','Sep','Oct','Nov','Dec']
#copy_air_visit_data_train.groupby(copy_air_visit_data_train['visit_date'].dt.weekday_name)['visitors'].median().reindex(days).plot(kind='bar', stacked=True, figsize=[15,6], colormap='gist_rainbow')

#copy_air_visit_data_train.groupby(copy_air_visit_data_train['Month'].astype('int').apply(lambda x: calendar.month_abbr[x]))['visitors'].median().reindex(months).plot(kind='bar', stacked=True, figsize=[15,6], colormap='Greens')
group_by_month = copy_air_visit_data_train.groupby(copy_air_visit_data_train['Month'].astype('int').apply(lambda x: calendar.month_abbr[x]))['visitors'].median().reindex(months)
group_by_month.plot(kind='bar', stacked=True, figsize=[15,6], colormap='jet')

print(GeoSeries([loads('POINT(1 2)'), loads('POINT(1.5 2.5)'), loads('POINT(2 3)')]))

geometry = [Polygon(x,y) for x, y in zip(copy_air_visit_data_train['longitude'],
            copy_air_visit_data_train['latitude'])]
gs = GeoSeries(geometry, index=copy_air_visit_data_train['air_store_id'])
print(gs)
gdf = gpd.GeoDataFrame(copy_air_visit_data_train,geometry=geometry)


#gdf.plot(cmap='Set2', figsize=(8, 8))
gdf.plot()
#print(gdf.crs)


#group_by_date.plot(kind='line', stacked=True, figsize=[15,6], colormap='winter')




