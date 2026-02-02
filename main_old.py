import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit


df = pd.read_csv('insurance (1).csv')
# print(df.duplicated().any())
df = df.drop_duplicates().reset_index(drop=True)
# print(df.duplicated().any())
 
df['bmi_category'] = pd.cut(
    df['bmi'],bins=[0,18.5,24.9,29.6,float('inf')],
            labels=['Underweight','Normal','Overweight','Obese']
)


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=40)

for train_index,test_index in split.split(df,df['bmi_category']):
    strat_train_set = df.loc[train_index].drop('bmi',axis=1)
    strat_test_set = df.loc[test_index].drop('bmi',axis=1)

data = strat_train_set.copy()
# data_lables = df['charges'].copy()
data_labels = strat_train_set['charges'].copy()

data = data.drop('charges',axis=1).copy()   
print(data,'data')

num_attribute = data.drop(['sex','smoker','region','bmi_category'], axis=1).columns.tolist()
print(num_attribute)
cat_sttribute = ['sex','smoker','region','bmi_category']
print(cat_sttribute)

num_pipline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

cat_pipline = Pipeline([
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

full_pipline = ColumnTransformer([
    ('num',num_pipline,num_attribute),
    ('cat',cat_pipline,cat_sttribute)
])

data_prepared = full_pipline.fit_transform(data)
print(data_prepared)

lin_reg = LinearRegression()
lin_reg.fit(data_prepared,data_labels)
lin_preds = lin_reg.predict(data_prepared)
lin_rmse = root_mean_squared_error(data_labels,lin_preds)
print("Linear",lin_rmse)

decTree_reg = DecisionTreeRegressor()
decTree_reg.fit(data_prepared,data_labels)
decTree_preds = decTree_reg.predict(data_prepared)
decTree_rmse = root_mean_squared_error(data_labels,decTree_preds)
print("decTree",lin_rmse)

Ran_reg = RandomForestRegressor()
Ran_reg.fit(data_prepared,data_labels)
Ran_predic = Ran_reg.predict(data_prepared)
Ran_rmse = root_mean_squared_error(data_labels,Ran_predic)
print("RanFo",Ran_rmse)