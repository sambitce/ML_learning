import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import calendar
'''
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry import Point
from shapely.wkt import loads
import plotly.plotly as py
from shapely.geometry import Polygon
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer ,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier

def load_data(data_file_name):
    return pd.read_csv(data_file_name)

def create_train_test_data(data):
    return train_test_split(data,
                            test_size=0.2,
                            random_state=42)
    
air_visit_data = load_data('air_visit_data.csv')
#print(air_visit_data.head())
#print(air_visit_data.info())


#print(air_visit_data["air_store_id"].value_counts())
#print(air_visit_data["visit_date"].value_counts())
#print(air_visit_data.describe())

air_store_info = load_data('air_store_info.csv')


copy_air_visit_data = air_visit_data.copy()
air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
#print("After")


#print(copy_air_visit_data.head())
#print(copy_air_visit_data.describe())
#print(copy_air_visit_data.info())

air_visit_data['Month'] = air_visit_data['visit_date'].dt.month
#print(air_visit_data['Month'])
#print("end")

air_visit_data.hist(bins=25 , figsize=(25,10))
plt.show()

air_visit_store_data = pd.merge(left=air_visit_data,
                                right=air_store_info,
                                left_on='air_store_id',
                                right_on='air_store_id' )

train_set , test_set = create_train_test_data(air_visit_store_data)

copy_air_visit_data_train = train_set.copy()
copy_air_visit_data_test = test_set.copy()


#print(copy_air_visit_data_train.head())



copy_air_visit_data_train.plot(kind="scatter" , x="latitude" , 
                              y="longitude",
                              alpha=0.4,
                              s=copy_air_visit_data_train["visitors"]*10,
                              label="Total visitors" ,
                              c="visitors" ,
                              cmap=plt.get_cmap("jet"),
                              colorbar=True,
                              figsize=(13,10))



fig, ax = plt.subplots(figsize=(15,7))
copy_air_visit_data_train.groupby('visit_date')['visitors'].sum().plot(ax=ax)

group_by_genre = copy_air_visit_data_train.groupby(['visit_date','air_genre_name']).sum()['visitors'].unstack()

group_by_genre.plot(kind='line', stacked=True, figsize=[15,6], colormap='gist_rainbow')

months =['Jan' ,'Feb','Mar','Apr','May','Jun',
         'Jul','Aug','Sep','Oct','Nov','Dec']


fig, ax = plt.subplots(figsize=(15,7))
mean_month_visitors = copy_air_visit_data_train.groupby(copy_air_visit_data_train['Month'].astype('int').apply(lambda x: calendar.month_abbr[x]))['visitors'].median().reindex(months).plot(kind='bar' ,colormap='gist_rainbow',ax=ax )

##(kind='bar',  figsize=[15,6], colormap='Greens')

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']


fig, ax = plt.subplots(figsize=(15,7))
mean_day_visitors = air_visit_data.groupby(air_visit_data['visit_date'].dt.weekday_name)['visitors'].median().reindex(days).plot(kind='bar', colormap='spring_r',ax=ax)


### below plot alone is not working...group by is working ####
#air_visit_data.groupby(air_visit_data['visit_date'].dt.weekday_name)['visitors'].median().reindex(days).plot(kind='bar', stacked=True, figsize=[15,6], colormap='gist_rainbow')


#copy_air_visit_data_train.groupby(copy_air_visit_data_train['Month'].astype('int').apply(lambda x: calendar.month_abbr[x]))['visitors'].median().reindex(months).plot(kind='bar', stacked=True, figsize=[15,6], colormap='Greens')
#group_by_month = copy_air_visit_data_train.groupby(copy_air_visit_data_train['Month'].astype('int').apply(lambda x: calendar.month_abbr[x]))['visitors'].median().reindex(months)
#group_by_month.plot(kind='bar', stacked=True, figsize=[15,6], colormap='jet')


'''
geometry = [Point(xy) for xy in zip(copy_air_visit_data_train['longitude'],
            copy_air_visit_data_train['latitude'])]
gs = GeoSeries(geometry, index=copy_air_visit_data_train['air_store_id'])

gdf = gpd.GeoDataFrame(copy_air_visit_data_train,geometry=geometry)


#gdf.plot(cmap='Set2', figsize=(8, 8))
gdf.plot()
#print(gdf.crs)
'''

copy_air_visit_store_data = pd.merge(left=copy_air_visit_data,
                                right=air_store_info,
                                left_on='air_store_id',
                                right_on='air_store_id' )

lbl = preprocessing.LabelEncoder()
'''
copy_air_visit_store_data=copy_air_visit_store_data.drop(['air_area_name'] ,axis=1)
copy_air_visit_store_data=copy_air_visit_store_data.drop(['air_genre_name'] ,axis=1)
'''

copy_air_visit_store_data['visit_date'] = pd.to_datetime(copy_air_visit_store_data['visit_date'])

copy_air_visit_store_data['Month'] = copy_air_visit_store_data['visit_date'].dt.month
copy_air_visit_store_data['Day'] = copy_air_visit_store_data['visit_date'].dt.day
copy_air_visit_store_data['WeekDay'] = copy_air_visit_store_data['visit_date'].dt.weekday
copy_air_visit_store_data['Year'] = copy_air_visit_store_data['visit_date'].dt.year
#copy_air_visit_store_data['WeekDay'] = copy_air_visit_store_data['visit_date'].dt.weekday
copy_air_visit_store_data['lat_long'] = copy_air_visit_store_data['latitude'] + copy_air_visit_store_data['longitude']

copy_air_visit_store_data['air_store_id'] = lbl.fit_transform(copy_air_visit_store_data['air_store_id'])

copy_air_visit_store_data['visit_date'] = lbl.fit_transform(copy_air_visit_store_data['visit_date'])



copy_air_visit_store_data['air_area_name'] = lbl.fit_transform(copy_air_visit_store_data['air_area_name'])

copy_air_visit_store_data['air_genre_name'] = lbl.fit_transform(copy_air_visit_store_data['air_genre_name'])

copy_air_visit_store_data = copy_air_visit_store_data.drop(['visit_date'],axis=1)


copy_air_visit_store_data.dropna(axis=0, how='all')

print("Going To Print" )
print("copy_air_visit_store_data" , copy_air_visit_store_data)
X_all =  copy_air_visit_store_data.drop(['visitors'],axis=1)

Y_all = copy_air_visit_store_data['visitors'].values



X_train, X_test,Y_train,Y_test = train_test_split(X_all,Y_all,test_size=0.20,random_state=23)



'''  SGD algorithm 

from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()

copy_air_visit_store_data['air_store_id'] = lbl.fit_transform(copy_air_visit_store_data['air_store_id'])

copy_air_visit_store_data['visit_date'] = lbl.fit_transform(copy_air_visit_store_data['visit_date'])

copy_air_visit_store_data.dropna(axis=0, how='all')

train_set , test_set = create_train_test_data(copy_air_visit_store_data)

train_set_features =  train_set.drop(['visitors'],axis=1)

train_set_labels = train_set['visitors']

test_features =  test_set.drop(['visitors'],axis=1)
test_labels = test_set['visitors']

test_set.to_csv('test_data.csv' ,header=True)

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

scaler = StandardScaler()
scaler.fit(train_set_features)

train_set_features = scaler.transform(train_set_features)
test_features = scaler.transform(test_features)

clf = linear_model.SGDRegressor(loss= "squared_loss" ,average=True)

#fitted = clf.fit(train_set_features,train_set_labels)

#predicted = clf.predict(test_features)



# create file
print('Generating submission file ...')
#results = pd.DataFrame({'Demanda_uni_equil': predicted}, dtype=int)


        
#Writting to csv
#results.to_csv('regressor.csv', index=True, header=True, index_label='id')
''' 
#### END OF SGD algorithm #####

############Random Forest classifier ##############

#### binary encoder 1 shot encoding learning rate multiplies 3 ,stratified random split 

model1 = RandomForestRegressor()
model2=tree.DecisionTreeRegressor()
model3=KNeighborsClassifier()

'''
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
'''
#acc_scorer = make_scorer(accuracy_score)

#grid_obj = GridSearchCV(clf,parameters, scoring=acc_scorer)

#grid_obj = grid_obj.fit(train_set_features,train_set_labels)

#clf = grid_obj.best_estimator_



model1.fit(X_train,Y_train)

import pickle 
pkl_filename="RandomForest.pkl"
with open(pkl_filename,'wb') as file:
    pickle.dump(model1,file)

temp_X_test = X_test[:1]
temp_X_test['Year']= 2015
temp_X_test['Month']= 2
temp_X_test['Date']=10
print("temp_x_Test" , temp_X_test)
temp_X_test.to_csv('X_test.csv')
predictions = model1.predict(X_test[:1])

#print("Accuracy score is" , accuracy_score(Y_test, predictions))

from sklearn.metrics import mean_squared_error

def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5

def plot_actual_predicted(actual, predicted):
    print('RMSE: ', RMSLE(actual, predicted))
    print("actual",actual)
    print("predicted",predicted)
    tmp = pd.DataFrame({'actual': actual, 'predicted': predicted}).sort_values(['actual'])
    plt.scatter(range(tmp.shape[0]), tmp['predicted'], color='green')
    plt.scatter(range(tmp.shape[0]), tmp['actual'], color='blue')
    plt.show()
    del tmp

plot_actual_predicted(Y_test[:1],predictions)
plt.scatter(Y_test[:1],predictions)


# create file
print('Generating submission file ...')
results = pd.DataFrame({'Year': X_test[:1]['Year'] ,'Month': X_test[:1]['Month'],'Day': X_test[:1]['Day'], 'Predicted': predictions, 'Actual' : Y_test[:1]})


        
#Writting to csv
results.to_csv('regressor.csv', index=True, header=True, index_label='id')



########################End RandomFOrest #########################
#group_by_date.plot(kind='line', stacked=True, figsize=[15,6], colormap='winter')



