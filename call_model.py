import pickle
import pandas as pd

def forecast(date1):
    with open ("RandomForest.pkl" ,'rb') as file:
        random_forest_model = pickle.load(file)
    
    def load_data(data_file_name):
        return pd.read_csv(data_file_name)
    
    
        
    air_visit_data = load_data('X_test.csv')
    year,month,day = date1.split("/")
    print(year, month,day)
    air_visit_data['Year'] =year
    air_visit_data['Month'] =month
    air_visit_data['Day'] = day
    
    air_visit_data=air_visit_data.drop(['Date'],axis=1 )
    air_visit_data=air_visit_data.drop(air_visit_data.columns[0],axis=1 )
    print(air_visit_data) 
    print(date1)
    predictions=random_forest_model.predict(air_visit_data)
    
    return str(predictions)
    

###output = forecast("2017/03/28")    
###print ("Output is" , str(output))
