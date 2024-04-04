import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 # import dataset 
data = pd.read_csv(r"C:\Users\pc\Desktop\BDAI1\fouille des donnees\LifeExpectancyData.csv")
data.head()

# shape of data :)
data.shape

# show columns names :)
data.columns

# we have to rename columns :) 
data.columns = ['Country', 'Year', 'Status', 'Life_expectancy', 'Adult_Mortality',
       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under_five deaths ', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
       'thinness_1_19_years', 'thinness_5_9_years',
       'Income_composition_of_resources', 'Schooling']

# Visualize categorical data for "Status"
sns.countplot(data['Status'])

# Extract numerical data into dataframe "numerical_data" to be insure "dataset normally distributed" :)
# numerical_data = data.select_dtypes(include=['float'])
# print(f"numerical_data_shape: {numerical_data.shape}")

data.select_dtypes(include=np.number)

def plot_feature_distribution(df,features): 
    i=0
    sns.set_style('whitegrid')
    plt.figure()
    fig ,ax = plt.subplots(4,4,figsize=(15,10))

    for feature in features: 
        i+=1 
        plt.subplot(5,5,i)
        sns.kdeplot(df[feature])
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
        
        
# Visualize numerical data :)
features = data.select_dtypes(include=np.number).columns
plot_feature_distribution(data.select_dtypes(include=np.number),features)

# # Looking for outliers :) 
# plt.figure(figsize=(20,5))
# data.select_dtypes(include=['float']).apply(np.log).boxplot()
# plt.xticks(rotation = 90)
        
## Looking for Correlations numerical feature with target column "Life_expectancy"
corr_matrix = data.corr()
corr_matrix["Life_expectancy"].sort_values(ascending=False)        


# Visualize Life Expectancy Over Years :)
sns.lineplot(x='Year',y='Life_expectancy',data=data)
plt.title("Life Expectancy Over Years",fontsize=13)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Life_expectancy',fontsize=12)


data.head()
# Check about Null values 
data.isna().sum()

# copy new dataframe :) 
data_ = data.dropna().copy()

data_.head()


# Check about Duplicate value :) 
data_.duplicated().all()
# there is no duplication value 


# Build PipeLine for machine learning algorithms "Transformation Pipelines" :) 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 

# Transform Numerical Data 
num_pipeline = make_pipeline(StandardScaler())
# Transform Categorical data 
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))


from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

# spliting data into X and (y => target data)
X = data_.drop('Life_expectancy',axis=1)
y = data_['Life_expectancy']

# Split data into training set and testing set :)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)
print(f"X_train: {X_train.shape}\nX_test: {X_test.shape}\ny_train: {y_train.shape}\ny_test: {y_test.shape}")


#linear regression

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(X_train, y_train)
 # predict new instance :) 
lin_reg_pred = lin_reg.predict(X_test)

# import mean_squared_error "using mean_squared_error to evaluate model" 
from sklearn.metrics import mean_squared_error
lin_rmse = mean_squared_error(y_test, lin_reg_pred,
                              squared=False)
lin_rmse
# Measure score for Linear regression model :) 
print(f"Linear_regression_score: {lin_reg.score(X_test,y_test)}")


# random forrest 
from sklearn.ensemble import RandomForestRegressor

RFR = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
RFR.fit(X_train, y_train)

RFR_rmse_pred = RFR.predict(X_test)
RFR_rmse = mean_squared_error(y_test, RFR_rmse_pred,
                              squared=False)
print(f"RandomForestRegressor_RMSE: {RFR_rmse}")

print(f"RandomForestRegressor_SCORE: {RFR.score(X_test, y_test)}")

#svr

# Super Vector Regressor "SVR"
from sklearn.svm import SVR 

# Build model 
SVR_= SVR(kernel='linear')
SVR_model = make_pipeline(preprocessing, SVR_)
SVR_model.fit(X_train, y_train)
SVR_model_pred = SVR_model.predict(X_test)
SVR_model_rmse = mean_squared_error(y_test, SVR_model_pred,
                              squared=False)
print(f"SVR_model_RMSE: {RFR_rmse}")
# Measure score for SVR model :) 
print(f"SVR_MODEL_SCORE: {SVR_model.score(X_train,y_train)}")
import sea