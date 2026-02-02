import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'


def build_pipeline(num_attribute,cat_sttribute):

    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('cat',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attribute),
        ('cat',cat_pipeline,cat_sttribute)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('insurance (1).csv')

    df = df.drop_duplicates().reset_index(drop=True)

    df['bmi_category'] = pd.cut(df['bmi'],bins=[0,18.5,24.9,29.6,float('inf')],
            labels=['Underweight','Normal','Overweight','Obese'])
    
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=40)

    for train_index,test_index in split.split(df,df['bmi_category']):
        # strat_test_set
        df.loc[test_index].drop('bmi',axis=1).to_csv('input.csv',index=False)
        strat_train_set = df.loc[train_index].drop('bmi',axis=1)
        

    # strat_test_set.drop('charges', axis=1).to_csv('input.csv', index=False)

    data_labels = strat_train_set['charges'].copy()

    data_feature = strat_train_set.drop('charges',axis=1)

    num_attribute = strat_train_set.drop(['sex','smoker','region','bmi_category','charges'],axis=1).columns.tolist()

    cat_attribute = ['sex','smoker','region','bmi_category']

    pipeline = build_pipeline(num_attribute,cat_attribute)
    data_prepared = pipeline.fit_transform(data_feature)
    
    model = RandomForestRegressor(random_state=40)

    model.fit(data_prepared,data_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
else:
    #lets do infrance
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')

    transformed_input = pipeline.transform(input_data)
    pridict = model.predict(transformed_input)  
    input_data['charges'] = pridict
    input_data.to_csv("output.csv" , index=False)
    print("infreance done")  
    