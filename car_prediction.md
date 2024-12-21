# importing libraries


```python
# Importing essential libraries
import numpy as np
import pandas as pd

# Visualization tools for exploratory data analysis (EDA)
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

# Evaluation metric libraries for model performance
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, mean_absolute_error,  accuracy_score,
    mean_squared_error, r2_score
)

# Preprocessing tools
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Libraries for splitting the dataset and hyperparameter tuning
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    RepeatedStratifiedKFold
)
from sklearn.model_selection import cross_val_score

# Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb  # Xtreme Gradient Boosting (XGBoost)

# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Inline plotting for Jupyter Notebooks
%matplotlib inline

```


```python
car=pd.read_csv(r"C:\Users\lakshita\Desktop\datasets\MLmodels\car_prediction\car_price_prediction.csv")
```


```python
car.head
```




    <bound method NDFrame.head of      car_ID  symboling                   CarName fueltype aspiration  \
    0         1          3        alfa-romero giulia      gas        std   
    1         2          3       alfa-romero stelvio      gas        std   
    2         3          1  alfa-romero Quadrifoglio      gas        std   
    3         4          2               audi 100 ls      gas        std   
    4         5          2                audi 100ls      gas        std   
    ..      ...        ...                       ...      ...        ...   
    200     201         -1           volvo 145e (sw)      gas        std   
    201     202         -1               volvo 144ea      gas      turbo   
    202     203         -1               volvo 244dl      gas        std   
    203     204         -1                 volvo 246   diesel      turbo   
    204     205         -1               volvo 264gl      gas      turbo   
    
        doornumber      carbody drivewheel enginelocation  wheelbase  ...  \
    0          two  convertible        rwd          front       88.6  ...   
    1          two  convertible        rwd          front       88.6  ...   
    2          two    hatchback        rwd          front       94.5  ...   
    3         four        sedan        fwd          front       99.8  ...   
    4         four        sedan        4wd          front       99.4  ...   
    ..         ...          ...        ...            ...        ...  ...   
    200       four        sedan        rwd          front      109.1  ...   
    201       four        sedan        rwd          front      109.1  ...   
    202       four        sedan        rwd          front      109.1  ...   
    203       four        sedan        rwd          front      109.1  ...   
    204       four        sedan        rwd          front      109.1  ...   
    
         enginesize  fuelsystem  boreratio  stroke compressionratio horsepower  \
    0           130        mpfi       3.47    2.68              9.0        111   
    1           130        mpfi       3.47    2.68              9.0        111   
    2           152        mpfi       2.68    3.47              9.0        154   
    3           109        mpfi       3.19    3.40             10.0        102   
    4           136        mpfi       3.19    3.40              8.0        115   
    ..          ...         ...        ...     ...              ...        ...   
    200         141        mpfi       3.78    3.15              9.5        114   
    201         141        mpfi       3.78    3.15              8.7        160   
    202         173        mpfi       3.58    2.87              8.8        134   
    203         145         idi       3.01    3.40             23.0        106   
    204         141        mpfi       3.78    3.15              9.5        114   
    
         peakrpm citympg  highwaympg    price  
    0       5000      21          27  13495.0  
    1       5000      21          27  16500.0  
    2       5000      19          26  16500.0  
    3       5500      24          30  13950.0  
    4       5500      18          22  17450.0  
    ..       ...     ...         ...      ...  
    200     5400      23          28  16845.0  
    201     5300      19          25  19045.0  
    202     5500      18          23  21485.0  
    203     4800      26          27  22470.0  
    204     5400      19          25  22625.0  
    
    [205 rows x 26 columns]>




```python
car.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205.000000</td>
      <td>...</td>
      <td>205.000000</td>
      <td>205</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>147</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>toyota corona</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>mpfi</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>185</td>
      <td>168</td>
      <td>115</td>
      <td>96</td>
      <td>120</td>
      <td>202</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>94</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>103.000000</td>
      <td>0.834146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.756585</td>
      <td>...</td>
      <td>126.907317</td>
      <td>NaN</td>
      <td>3.329756</td>
      <td>3.255415</td>
      <td>10.142537</td>
      <td>104.117073</td>
      <td>5125.121951</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>13276.710571</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59.322565</td>
      <td>1.245307</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.021776</td>
      <td>...</td>
      <td>41.642693</td>
      <td>NaN</td>
      <td>0.270844</td>
      <td>0.313597</td>
      <td>3.972040</td>
      <td>39.544167</td>
      <td>476.985643</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>7988.852332</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.600000</td>
      <td>...</td>
      <td>61.000000</td>
      <td>NaN</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.500000</td>
      <td>...</td>
      <td>97.000000</td>
      <td>NaN</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7788.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>103.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97.000000</td>
      <td>...</td>
      <td>120.000000</td>
      <td>NaN</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5200.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>154.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102.400000</td>
      <td>...</td>
      <td>141.000000</td>
      <td>NaN</td>
      <td>3.580000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16503.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>205.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.900000</td>
      <td>...</td>
      <td>326.000000</td>
      <td>NaN</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 26 columns</p>
</div>




```python
car.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   car_ID            205 non-null    int64  
     1   symboling         205 non-null    int64  
     2   CarName           205 non-null    object 
     3   fueltype          205 non-null    object 
     4   aspiration        205 non-null    object 
     5   doornumber        205 non-null    object 
     6   carbody           205 non-null    object 
     7   drivewheel        205 non-null    object 
     8   enginelocation    205 non-null    object 
     9   wheelbase         205 non-null    float64
     10  carlength         205 non-null    float64
     11  carwidth          205 non-null    float64
     12  carheight         205 non-null    float64
     13  curbweight        205 non-null    int64  
     14  enginetype        205 non-null    object 
     15  cylindernumber    205 non-null    object 
     16  enginesize        205 non-null    int64  
     17  fuelsystem        205 non-null    object 
     18  boreratio         205 non-null    float64
     19  stroke            205 non-null    float64
     20  compressionratio  205 non-null    float64
     21  horsepower        205 non-null    int64  
     22  peakrpm           205 non-null    int64  
     23  citympg           205 non-null    int64  
     24  highwaympg        205 non-null    int64  
     25  price             205 non-null    float64
    dtypes: float64(8), int64(8), object(10)
    memory usage: 41.8+ KB
    


```python
print(car.dtypes)
```

    car_ID                int64
    symboling             int64
    CarName              object
    fueltype             object
    aspiration           object
    doornumber           object
    carbody              object
    drivewheel           object
    enginelocation       object
    wheelbase           float64
    carlength           float64
    carwidth            float64
    carheight           float64
    curbweight            int64
    enginetype           object
    cylindernumber       object
    enginesize            int64
    fuelsystem           object
    boreratio           float64
    stroke              float64
    compressionratio    float64
    horsepower            int64
    peakrpm               int64
    citympg               int64
    highwaympg            int64
    price               float64
    dtype: object
    


```python
car.value_counts
```




    <bound method DataFrame.value_counts of      car_ID  symboling                   CarName fueltype aspiration  \
    0         1          3        alfa-romero giulia      gas        std   
    1         2          3       alfa-romero stelvio      gas        std   
    2         3          1  alfa-romero Quadrifoglio      gas        std   
    3         4          2               audi 100 ls      gas        std   
    4         5          2                audi 100ls      gas        std   
    ..      ...        ...                       ...      ...        ...   
    200     201         -1           volvo 145e (sw)      gas        std   
    201     202         -1               volvo 144ea      gas      turbo   
    202     203         -1               volvo 244dl      gas        std   
    203     204         -1                 volvo 246   diesel      turbo   
    204     205         -1               volvo 264gl      gas      turbo   
    
        doornumber      carbody drivewheel enginelocation  wheelbase  ...  \
    0          two  convertible        rwd          front       88.6  ...   
    1          two  convertible        rwd          front       88.6  ...   
    2          two    hatchback        rwd          front       94.5  ...   
    3         four        sedan        fwd          front       99.8  ...   
    4         four        sedan        4wd          front       99.4  ...   
    ..         ...          ...        ...            ...        ...  ...   
    200       four        sedan        rwd          front      109.1  ...   
    201       four        sedan        rwd          front      109.1  ...   
    202       four        sedan        rwd          front      109.1  ...   
    203       four        sedan        rwd          front      109.1  ...   
    204       four        sedan        rwd          front      109.1  ...   
    
         enginesize  fuelsystem  boreratio  stroke compressionratio horsepower  \
    0           130        mpfi       3.47    2.68              9.0        111   
    1           130        mpfi       3.47    2.68              9.0        111   
    2           152        mpfi       2.68    3.47              9.0        154   
    3           109        mpfi       3.19    3.40             10.0        102   
    4           136        mpfi       3.19    3.40              8.0        115   
    ..          ...         ...        ...     ...              ...        ...   
    200         141        mpfi       3.78    3.15              9.5        114   
    201         141        mpfi       3.78    3.15              8.7        160   
    202         173        mpfi       3.58    2.87              8.8        134   
    203         145         idi       3.01    3.40             23.0        106   
    204         141        mpfi       3.78    3.15              9.5        114   
    
         peakrpm citympg  highwaympg    price  
    0       5000      21          27  13495.0  
    1       5000      21          27  16500.0  
    2       5000      19          26  16500.0  
    3       5500      24          30  13950.0  
    4       5500      18          22  17450.0  
    ..       ...     ...         ...      ...  
    200     5400      23          28  16845.0  
    201     5300      19          25  19045.0  
    202     5500      18          23  21485.0  
    203     4800      26          27  22470.0  
    204     5400      19          25  22625.0  
    
    [205 rows x 26 columns]>




```python
car.sort_values
```




    <bound method DataFrame.sort_values of      car_ID  symboling                   CarName fueltype aspiration  \
    0         1          3        alfa-romero giulia      gas        std   
    1         2          3       alfa-romero stelvio      gas        std   
    2         3          1  alfa-romero Quadrifoglio      gas        std   
    3         4          2               audi 100 ls      gas        std   
    4         5          2                audi 100ls      gas        std   
    ..      ...        ...                       ...      ...        ...   
    200     201         -1           volvo 145e (sw)      gas        std   
    201     202         -1               volvo 144ea      gas      turbo   
    202     203         -1               volvo 244dl      gas        std   
    203     204         -1                 volvo 246   diesel      turbo   
    204     205         -1               volvo 264gl      gas      turbo   
    
        doornumber      carbody drivewheel enginelocation  wheelbase  ...  \
    0          two  convertible        rwd          front       88.6  ...   
    1          two  convertible        rwd          front       88.6  ...   
    2          two    hatchback        rwd          front       94.5  ...   
    3         four        sedan        fwd          front       99.8  ...   
    4         four        sedan        4wd          front       99.4  ...   
    ..         ...          ...        ...            ...        ...  ...   
    200       four        sedan        rwd          front      109.1  ...   
    201       four        sedan        rwd          front      109.1  ...   
    202       four        sedan        rwd          front      109.1  ...   
    203       four        sedan        rwd          front      109.1  ...   
    204       four        sedan        rwd          front      109.1  ...   
    
         enginesize  fuelsystem  boreratio  stroke compressionratio horsepower  \
    0           130        mpfi       3.47    2.68              9.0        111   
    1           130        mpfi       3.47    2.68              9.0        111   
    2           152        mpfi       2.68    3.47              9.0        154   
    3           109        mpfi       3.19    3.40             10.0        102   
    4           136        mpfi       3.19    3.40              8.0        115   
    ..          ...         ...        ...     ...              ...        ...   
    200         141        mpfi       3.78    3.15              9.5        114   
    201         141        mpfi       3.78    3.15              8.7        160   
    202         173        mpfi       3.58    2.87              8.8        134   
    203         145         idi       3.01    3.40             23.0        106   
    204         141        mpfi       3.78    3.15              9.5        114   
    
         peakrpm citympg  highwaympg    price  
    0       5000      21          27  13495.0  
    1       5000      21          27  16500.0  
    2       5000      19          26  16500.0  
    3       5500      24          30  13950.0  
    4       5500      18          22  17450.0  
    ..       ...     ...         ...      ...  
    200     5400      23          28  16845.0  
    201     5300      19          25  19045.0  
    202     5500      18          23  21485.0  
    203     4800      26          27  22470.0  
    204     5400      19          25  22625.0  
    
    [205 rows x 26 columns]>




```python
car.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>201</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>202</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>203</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>204</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 26 columns</p>
</div>




```python
car.nunique()
```




    car_ID              205
    symboling             6
    CarName             147
    fueltype              2
    aspiration            2
    doornumber            2
    carbody               5
    drivewheel            3
    enginelocation        2
    wheelbase            53
    carlength            75
    carwidth             44
    carheight            49
    curbweight          171
    enginetype            7
    cylindernumber        7
    enginesize           44
    fuelsystem            8
    boreratio            38
    stroke               37
    compressionratio     32
    horsepower           59
    peakrpm              23
    citympg              29
    highwaympg           30
    price               189
    dtype: int64




```python
print(car['CarName'].unique())
```

    ['alfa-romero giulia' 'alfa-romero stelvio' 'alfa-romero Quadrifoglio'
     'audi 100 ls' 'audi 100ls' 'audi fox' 'audi 5000' 'audi 4000'
     'audi 5000s (diesel)' 'bmw 320i' 'bmw x1' 'bmw x3' 'bmw z4' 'bmw x4'
     'bmw x5' 'chevrolet impala' 'chevrolet monte carlo' 'chevrolet vega 2300'
     'dodge rampage' 'dodge challenger se' 'dodge d200' 'dodge monaco (sw)'
     'dodge colt hardtop' 'dodge colt (sw)' 'dodge coronet custom'
     'dodge dart custom' 'dodge coronet custom (sw)' 'honda civic'
     'honda civic cvcc' 'honda accord cvcc' 'honda accord lx'
     'honda civic 1500 gl' 'honda accord' 'honda civic 1300' 'honda prelude'
     'honda civic (auto)' 'isuzu MU-X' 'isuzu D-Max ' 'isuzu D-Max V-Cross'
     'jaguar xj' 'jaguar xf' 'jaguar xk' 'maxda rx3' 'maxda glc deluxe'
     'mazda rx2 coupe' 'mazda rx-4' 'mazda glc deluxe' 'mazda 626' 'mazda glc'
     'mazda rx-7 gs' 'mazda glc 4' 'mazda glc custom l' 'mazda glc custom'
     'buick electra 225 custom' 'buick century luxus (sw)' 'buick century'
     'buick skyhawk' 'buick opel isuzu deluxe' 'buick skylark'
     'buick century special' 'buick regal sport coupe (turbo)'
     'mercury cougar' 'mitsubishi mirage' 'mitsubishi lancer'
     'mitsubishi outlander' 'mitsubishi g4' 'mitsubishi mirage g4'
     'mitsubishi montero' 'mitsubishi pajero' 'Nissan versa' 'nissan gt-r'
     'nissan rogue' 'nissan latio' 'nissan titan' 'nissan leaf' 'nissan juke'
     'nissan note' 'nissan clipper' 'nissan nv200' 'nissan dayz' 'nissan fuga'
     'nissan otti' 'nissan teana' 'nissan kicks' 'peugeot 504' 'peugeot 304'
     'peugeot 504 (sw)' 'peugeot 604sl' 'peugeot 505s turbo diesel'
     'plymouth fury iii' 'plymouth cricket' 'plymouth satellite custom (sw)'
     'plymouth fury gran sedan' 'plymouth valiant' 'plymouth duster'
     'porsche macan' 'porcshce panamera' 'porsche cayenne' 'porsche boxter'
     'renault 12tl' 'renault 5 gtl' 'saab 99e' 'saab 99le' 'saab 99gle'
     'subaru' 'subaru dl' 'subaru brz' 'subaru baja' 'subaru r1' 'subaru r2'
     'subaru trezia' 'subaru tribeca' 'toyota corona mark ii' 'toyota corona'
     'toyota corolla 1200' 'toyota corona hardtop' 'toyota corolla 1600 (sw)'
     'toyota carina' 'toyota mark ii' 'toyota corolla'
     'toyota corolla liftback' 'toyota celica gt liftback'
     'toyota corolla tercel' 'toyota corona liftback' 'toyota starlet'
     'toyota tercel' 'toyota cressida' 'toyota celica gt' 'toyouta tercel'
     'vokswagen rabbit' 'volkswagen 1131 deluxe sedan' 'volkswagen model 111'
     'volkswagen type 3' 'volkswagen 411 (sw)' 'volkswagen super beetle'
     'volkswagen dasher' 'vw dasher' 'vw rabbit' 'volkswagen rabbit'
     'volkswagen rabbit custom' 'volvo 145e (sw)' 'volvo 144ea' 'volvo 244dl'
     'volvo 245' 'volvo 264gl' 'volvo diesel' 'volvo 246']
    


```python
duplicated_rows = car[car.duplicated()]
print(f"duplicated values are : {duplicated_rows.shape[0]}")
print(car.isnull().sum())
```

    duplicated values are : 0
    car_ID              0
    symboling           0
    CarName             0
    fueltype            0
    aspiration          0
    doornumber          0
    carbody             0
    drivewheel          0
    enginelocation      0
    wheelbase           0
    carlength           0
    carwidth            0
    carheight           0
    curbweight          0
    enginetype          0
    cylindernumber      0
    enginesize          0
    fuelsystem          0
    boreratio           0
    stroke              0
    compressionratio    0
    horsepower          0
    peakrpm             0
    citympg             0
    highwaympg          0
    price               0
    dtype: int64
    


```python
sns.pairplot(car)
```




    <seaborn.axisgrid.PairGrid at 0x284b6e77680>




    
![png](output_13_1.png)
    


# Graph b/w Carname and price


```python
print(car.head())
plt.figure(figsize=(10,30)) 
sns.scatterplot(x='price',y='CarName', data=car, color='grey') 
plt.title('Scatterplot of CarName V/S Price')
plt.xlabel('Price')
plt.ylabel('CarName')
plt.show()
```

       car_ID  symboling                   CarName fueltype aspiration doornumber  \
    0       1          3        alfa-romero giulia      gas        std        two   
    1       2          3       alfa-romero stelvio      gas        std        two   
    2       3          1  alfa-romero Quadrifoglio      gas        std        two   
    3       4          2               audi 100 ls      gas        std       four   
    4       5          2                audi 100ls      gas        std       four   
    
           carbody drivewheel enginelocation  wheelbase  ...  enginesize  \
    0  convertible        rwd          front       88.6  ...         130   
    1  convertible        rwd          front       88.6  ...         130   
    2    hatchback        rwd          front       94.5  ...         152   
    3        sedan        fwd          front       99.8  ...         109   
    4        sedan        4wd          front       99.4  ...         136   
    
       fuelsystem  boreratio  stroke compressionratio horsepower  peakrpm citympg  \
    0        mpfi       3.47    2.68              9.0        111     5000      21   
    1        mpfi       3.47    2.68              9.0        111     5000      21   
    2        mpfi       2.68    3.47              9.0        154     5000      19   
    3        mpfi       3.19    3.40             10.0        102     5500      24   
    4        mpfi       3.19    3.40              8.0        115     5500      18   
    
       highwaympg    price  
    0          27  13495.0  
    1          27  16500.0  
    2          26  16500.0  
    3          30  13950.0  
    4          22  17450.0  
    
    [5 rows x 26 columns]
    


    
![png](output_15_1.png)
    



```python
target_variable='price'
```


```python
px.box(car.carlength,car.carwidth)
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.32.0
* Copyright 2012-2024, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>




<div>                            <div id="44ecdf00-d8c0-46d7-a64f-b455cca7accc" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("44ecdf00-d8c0-46d7-a64f-b455cca7accc")) {                    Plotly.newPlot(                        "44ecdf00-d8c0-46d7-a64f-b455cca7accc",                        [{"alignmentgroup":"True","hovertemplate":"x=%{x}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#636efa"},"name":"","notched":false,"offsetgroup":"","orientation":"h","showlegend":false,"x":[64.1,64.1,65.5,66.2,66.4,66.3,71.4,71.4,71.4,67.9,64.8,64.8,64.8,64.8,66.9,66.9,67.9,70.9,60.3,63.6,63.6,63.8,63.8,63.8,63.8,63.8,63.8,63.8,64.6,66.3,63.9,63.9,64.0,64.0,64.0,64.0,63.9,65.2,65.2,65.2,62.5,65.2,66.0,61.8,63.6,63.6,65.2,69.6,69.6,70.6,64.2,64.2,64.2,64.2,64.2,65.7,65.7,65.7,65.7,66.5,66.5,66.5,66.5,66.5,66.5,66.1,66.1,70.3,70.3,70.3,71.7,71.7,70.5,71.7,72.0,68.0,64.4,64.4,64.4,63.8,65.4,65.4,66.3,66.3,66.3,65.4,65.4,65.4,65.4,63.8,63.8,63.8,63.8,63.8,63.8,63.8,63.8,63.8,63.8,65.2,65.2,66.5,66.5,66.5,67.9,67.9,67.9,68.4,68.4,68.4,68.4,68.4,68.4,68.4,68.4,68.4,68.4,68.3,63.8,63.8,63.8,63.8,63.8,64.6,66.3,68.3,65.0,65.0,65.0,72.3,66.5,66.6,66.5,66.5,66.5,66.5,66.5,66.5,63.4,63.6,63.8,65.4,65.4,65.4,65.4,65.4,65.4,65.4,65.4,65.4,63.6,63.6,63.6,63.6,63.6,63.6,64.4,64.4,64.4,64.4,64.4,64.4,64.4,64.0,64.0,64.0,64.0,65.6,65.6,65.6,65.6,65.6,65.6,66.5,66.5,66.5,66.5,66.5,67.7,67.7,66.5,66.5,65.5,65.5,65.5,65.5,65.5,65.5,65.5,64.2,64.0,66.9,66.9,66.9,67.2,67.2,67.2,67.2,67.2,67.2,68.9,68.8,68.9,68.9,68.9],"x0":" ","xaxis":"x","y0":" ","yaxis":"y","type":"box"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0]},"legend":{"tracegroupgap":0},"margin":{"t":60},"boxmode":"group"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('44ecdf00-d8c0-46d7-a64f-b455cca7accc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
plt.figure(figsize=(15, 8))  # Set the size of the plot
car_counts = car['CarName'].value_counts().head(20)  # Show top 20 car names
sns.barplot(x=car_counts.index, y=car_counts.values, palette='Oranges')
plt.title('Top 20 Cars by Sales')
plt.xlabel('Car Name')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

```


    
![png](output_18_0.png)
    



```python
sns.histplot(car['price'], kde=True, bins=30, color='violet')
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_19_0.png)
    



```python
sns.scatterplot(data=car, x='enginesize', y='price', hue='fueltype')
plt.title('Engine Size vs Price')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.show()
```


    
![png](output_20_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=car, x='carbody', y='price', palette='Set2')
plt.title('Price by Car Body Type')
plt.xlabel('Car Body Type')
plt.ylabel('Price')
plt.show()
```


    
![png](output_21_0.png)
    



```python
sns.pairplot(car[['price', 'enginesize', 'horsepower', 'curbweight']])
plt.suptitle('Pair Plot of Key Features', y=1.02)
plt.show()
```


    
![png](output_22_0.png)
    



```python
plt.figure(figsize=(19, 8))
sns.lineplot(data=car, x='horsepower', y='price', marker='o')
plt.title('Horsepower vs Price')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.show()
```


    
![png](output_23_0.png)
    



```python
sns.countplot(data=car, x='fueltype', palette='muted')
plt.title('Count of Cars by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()
```


    
![png](output_24_0.png)
    



```python
fig = px.scatter(car,x='enginesize',y='price',color='fueltype',title='Engine Size vs Price',labels={'enginesize': 'Engine Size', 'price': 'Price'})
fig.update_layout(title_font_size=18,xaxis_title='Engine Size',yaxis_title='Price',height=600,width=800)
fig.show()
```


<div>                            <div id="e5fb768d-f098-47ef-b018-d7ab628f5a13" class="plotly-graph-div" style="height:600px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e5fb768d-f098-47ef-b018-d7ab628f5a13")) {                    Plotly.newPlot(                        "e5fb768d-f098-47ef-b018-d7ab628f5a13",                        [{"hovertemplate":"fueltype=gas\u003cbr\u003eEngine Size=%{x}\u003cbr\u003ePrice=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"gas","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"gas","orientation":"v","showlegend":true,"x":[130,130,152,109,136,136,136,136,131,131,108,108,164,164,164,209,209,209,61,90,90,90,90,98,90,90,90,98,122,156,92,92,79,92,92,92,92,110,110,110,110,110,110,111,90,90,119,258,258,326,91,91,91,91,91,70,70,70,80,122,122,122,122,122,140,234,234,308,304,140,92,92,92,98,110,122,156,156,156,122,122,110,110,97,97,97,97,97,97,97,97,97,120,120,181,181,181,181,181,181,120,120,120,120,120,134,90,98,90,90,98,122,156,151,194,194,194,203,132,132,121,121,121,121,121,121,97,108,108,108,108,108,108,108,108,108,108,108,92,92,92,92,92,92,98,98,98,98,98,98,98,98,98,146,146,146,146,146,146,122,122,122,122,171,171,171,161,109,109,109,109,109,109,136,109,141,141,141,141,130,130,141,141,173,141],"xaxis":"x","y":[13495.0,16500.0,16500.0,13950.0,17450.0,15250.0,17710.0,18920.0,23875.0,17859.167,16430.0,16925.0,20970.0,21105.0,24565.0,30760.0,41315.0,36880.0,5151.0,6295.0,6575.0,5572.0,6377.0,7957.0,6229.0,6692.0,7609.0,8558.0,8921.0,12964.0,6479.0,6855.0,5399.0,6529.0,7129.0,7295.0,7295.0,7895.0,9095.0,8845.0,10295.0,12945.0,10345.0,6785.0,8916.5,8916.5,11048.0,32250.0,35550.0,36000.0,5195.0,6095.0,6795.0,6695.0,7395.0,10945.0,11845.0,13645.0,15645.0,8845.0,8495.0,10595.0,10245.0,11245.0,18280.0,34184.0,35056.0,40960.0,45400.0,16503.0,5389.0,6189.0,6669.0,7689.0,9959.0,8499.0,12629.0,14869.0,14489.0,6989.0,8189.0,9279.0,9279.0,5499.0,6649.0,6849.0,7349.0,7299.0,7799.0,7499.0,7999.0,8249.0,8949.0,9549.0,13499.0,14399.0,13499.0,17199.0,19699.0,18399.0,11900.0,12440.0,15580.0,16695.0,16630.0,18150.0,5572.0,7957.0,6229.0,6692.0,7609.0,8921.0,12764.0,22018.0,32528.0,34028.0,37028.0,31400.5,9295.0,9895.0,11850.0,12170.0,15040.0,15510.0,18150.0,18620.0,5118.0,7053.0,7603.0,7126.0,7775.0,9960.0,9233.0,11259.0,7463.0,10198.0,8013.0,11694.0,5348.0,6338.0,6488.0,6918.0,7898.0,8778.0,6938.0,7198.0,7738.0,8358.0,9258.0,8058.0,8238.0,9298.0,9538.0,8449.0,9639.0,9989.0,11199.0,11549.0,17669.0,8948.0,9988.0,10898.0,11248.0,16558.0,15998.0,15690.0,15750.0,7975.0,8195.0,8495.0,9995.0,11595.0,9980.0,13295.0,12290.0,12940.0,13415.0,15985.0,16515.0,18420.0,18950.0,16845.0,19045.0,21485.0,22625.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"fueltype=diesel\u003cbr\u003eEngine Size=%{x}\u003cbr\u003ePrice=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"diesel","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"diesel","orientation":"v","showlegend":true,"x":[122,134,183,183,183,183,103,152,152,152,152,152,110,110,110,97,97,97,97,145],"xaxis":"x","y":[10795.0,18344.0,25552.0,28248.0,28176.0,31600.0,7099.0,13200.0,13860.0,16900.0,17075.0,17950.0,7898.0,7788.0,10698.0,7775.0,7995.0,9495.0,13845.0,22470.0],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Engine Size"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Price"}},"legend":{"title":{"text":"fueltype"},"tracegroupgap":0},"title":{"text":"Engine Size vs Price","font":{"size":18}},"height":600,"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e5fb768d-f098-47ef-b018-d7ab628f5a13');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=car['highwaympg'], y=car['price'], color='blue', alpha=0.7)
plt.title('Highway MPG vs Price', fontsize=16)
plt.xlabel('Highway MPG', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid(True)
plt.show()
```


    
![png](output_26_0.png)
    



```python
correlation = car['highwaympg'].corr(car['price'])
print(f"Correlation between highwaympg and price: {correlation}")
```

    Correlation between highwaympg and price: -0.6975990916465566
    

#  Exclude Non-Numeric Columns


```python
print(car.columns)
```

    Index(['car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration',
           'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
           'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
           'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
           'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg',
           'price'],
          dtype='object')
    


```python
car = car.drop(['CarName', 'car_ID'], axis=1)
```


```python
# Select only numeric columns
numeric_data = car.select_dtypes(include=['float64', 'int64'])
# Compute the correlation matrix
correlation_matrix = numeric_data.corr()
# Display the correlation matrix
print(correlation_matrix)
```

                      symboling  wheelbase  carlength  carwidth  carheight  \
    symboling          1.000000  -0.531954  -0.357612 -0.232919  -0.541038   
    wheelbase         -0.531954   1.000000   0.874587  0.795144   0.589435   
    carlength         -0.357612   0.874587   1.000000  0.841118   0.491029   
    carwidth          -0.232919   0.795144   0.841118  1.000000   0.279210   
    carheight         -0.541038   0.589435   0.491029  0.279210   1.000000   
    curbweight        -0.227691   0.776386   0.877728  0.867032   0.295572   
    enginesize        -0.105790   0.569329   0.683360  0.735433   0.067149   
    boreratio         -0.130051   0.488750   0.606454  0.559150   0.171071   
    stroke            -0.008735   0.160959   0.129533  0.182942  -0.055307   
    compressionratio  -0.178515   0.249786   0.158414  0.181129   0.261214   
    horsepower         0.070873   0.353294   0.552623  0.640732  -0.108802   
    peakrpm            0.273606  -0.360469  -0.287242 -0.220012  -0.320411   
    citympg           -0.035823  -0.470414  -0.670909 -0.642704  -0.048640   
    highwaympg         0.034606  -0.544082  -0.704662 -0.677218  -0.107358   
    price             -0.079978   0.577816   0.682920  0.759325   0.119336   
    
                      curbweight  enginesize  boreratio    stroke  \
    symboling          -0.227691   -0.105790  -0.130051 -0.008735   
    wheelbase           0.776386    0.569329   0.488750  0.160959   
    carlength           0.877728    0.683360   0.606454  0.129533   
    carwidth            0.867032    0.735433   0.559150  0.182942   
    carheight           0.295572    0.067149   0.171071 -0.055307   
    curbweight          1.000000    0.850594   0.648480  0.168790   
    enginesize          0.850594    1.000000   0.583774  0.203129   
    boreratio           0.648480    0.583774   1.000000 -0.055909   
    stroke              0.168790    0.203129  -0.055909  1.000000   
    compressionratio    0.151362    0.028971   0.005197  0.186110   
    horsepower          0.750739    0.809769   0.573677  0.080940   
    peakrpm            -0.266243   -0.244660  -0.254976 -0.067964   
    citympg            -0.757414   -0.653658  -0.584532 -0.042145   
    highwaympg         -0.797465   -0.677470  -0.587012 -0.043931   
    price               0.835305    0.874145   0.553173  0.079443   
    
                      compressionratio  horsepower   peakrpm   citympg  \
    symboling                -0.178515    0.070873  0.273606 -0.035823   
    wheelbase                 0.249786    0.353294 -0.360469 -0.470414   
    carlength                 0.158414    0.552623 -0.287242 -0.670909   
    carwidth                  0.181129    0.640732 -0.220012 -0.642704   
    carheight                 0.261214   -0.108802 -0.320411 -0.048640   
    curbweight                0.151362    0.750739 -0.266243 -0.757414   
    enginesize                0.028971    0.809769 -0.244660 -0.653658   
    boreratio                 0.005197    0.573677 -0.254976 -0.584532   
    stroke                    0.186110    0.080940 -0.067964 -0.042145   
    compressionratio          1.000000   -0.204326 -0.435741  0.324701   
    horsepower               -0.204326    1.000000  0.131073 -0.801456   
    peakrpm                  -0.435741    0.131073  1.000000 -0.113544   
    citympg                   0.324701   -0.801456 -0.113544  1.000000   
    highwaympg                0.265201   -0.770544 -0.054275  0.971337   
    price                     0.067984    0.808139 -0.085267 -0.685751   
    
                      highwaympg     price  
    symboling           0.034606 -0.079978  
    wheelbase          -0.544082  0.577816  
    carlength          -0.704662  0.682920  
    carwidth           -0.677218  0.759325  
    carheight          -0.107358  0.119336  
    curbweight         -0.797465  0.835305  
    enginesize         -0.677470  0.874145  
    boreratio          -0.587012  0.553173  
    stroke             -0.043931  0.079443  
    compressionratio    0.265201  0.067984  
    horsepower         -0.770544  0.808139  
    peakrpm            -0.054275 -0.085267  
    citympg             0.971337 -0.685751  
    highwaympg          1.000000 -0.697599  
    price              -0.697599  1.000000  
    

# Encode Categorical Columns


```python
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                       'drivewheel', 'enginelocation', 'enginetype', 
                       'cylindernumber', 'fuelsystem']
for col in categorical_columns:
    car[col] = LabelEncoder().fit_transform(car[col])
car = pd.get_dummies(car, columns=['fueltype', 'aspiration', 'doornumber', 
                                   'carbody', 'drivewheel', 'enginelocation', 
                                   'enginetype', 'cylindernumber', 'fuelsystem'], drop_first=True)
```


```python
correlation_matrix = car.corr()
print(correlation_matrix)

```

                      symboling  wheelbase  carlength  carwidth  carheight  \
    symboling          1.000000  -0.531954  -0.357612 -0.232919  -0.541038   
    wheelbase         -0.531954   1.000000   0.874587  0.795144   0.589435   
    carlength         -0.357612   0.874587   1.000000  0.841118   0.491029   
    carwidth          -0.232919   0.795144   0.841118  1.000000   0.279210   
    carheight         -0.541038   0.589435   0.491029  0.279210   1.000000   
    curbweight        -0.227691   0.776386   0.877728  0.867032   0.295572   
    enginesize        -0.105790   0.569329   0.683360  0.735433   0.067149   
    boreratio         -0.130051   0.488750   0.606454  0.559150   0.171071   
    stroke            -0.008735   0.160959   0.129533  0.182942  -0.055307   
    compressionratio  -0.178515   0.249786   0.158414  0.181129   0.261214   
    horsepower         0.070873   0.353294   0.552623  0.640732  -0.108802   
    peakrpm            0.273606  -0.360469  -0.287242 -0.220012  -0.320411   
    citympg           -0.035823  -0.470414  -0.670909 -0.642704  -0.048640   
    highwaympg         0.034606  -0.544082  -0.704662 -0.677218  -0.107358   
    price             -0.079978   0.577816   0.682920  0.759325   0.119336   
    fueltype_1         0.194311  -0.308346  -0.212679 -0.233880  -0.284631   
    aspiration_1      -0.059866   0.257611   0.234539  0.300567   0.087311   
    doornumber_1       0.664073  -0.447357  -0.398568 -0.207168  -0.552208   
    carbody_1          0.168845  -0.008608   0.047292  0.066360  -0.072328   
    carbody_2          0.435648  -0.386094  -0.436269 -0.222308  -0.477476   
    carbody_3         -0.378341   0.291086   0.269647  0.154637   0.235863   
    carbody_4         -0.298243   0.210899   0.219683  0.060639   0.459148   
    drivewheel_1       0.102839  -0.460355  -0.508714 -0.472116  -0.100273   
    drivewheel_2      -0.076381   0.498830   0.538370  0.511149   0.039814   
    enginelocation_1   0.212471  -0.187790  -0.050989 -0.051698  -0.106234   
    enginetype_1       0.009347  -0.004156   0.009391  0.209136  -0.092628   
    enginetype_2      -0.133979   0.399603   0.261715  0.210771   0.319687   
    enginetype_3      -0.082855  -0.204037  -0.274413 -0.286211   0.036260   
    enginetype_4       0.037513  -0.183195  -0.118320 -0.124446  -0.046670   
    enginetype_5      -0.013597   0.166152   0.244053  0.348869  -0.065063   
    enginetype_6       0.245950  -0.081174  -0.057877 -0.013699  -0.238720   
    cylindernumber_1  -0.090188   0.261182   0.259894  0.397690   0.152982   
    cylindernumber_2  -0.034161  -0.309492  -0.400210 -0.523135   0.059696   
    cylindernumber_3  -0.000238   0.145842   0.262981  0.209246  -0.049777   
    cylindernumber_4   0.065707  -0.120709  -0.187445 -0.183473  -0.015076   
    cylindernumber_5  -0.047012   0.037803   0.100413  0.153516  -0.170181   
    cylindernumber_6   0.245950  -0.081174  -0.057877 -0.013699  -0.238720   
    fuelsystem_1      -0.034069  -0.396505  -0.487237 -0.522594  -0.079418   
    fuelsystem_2       0.212471  -0.070124  -0.049998 -0.011834  -0.206225   
    fuelsystem_3      -0.194311   0.308346   0.212679  0.233880   0.284631   
    fuelsystem_4       0.122067  -0.033294  -0.004831  0.012832  -0.101245   
    fuelsystem_5       0.012532   0.348891   0.511374  0.461896   0.108685   
    fuelsystem_6       0.181939  -0.117359  -0.079790 -0.046399  -0.278615   
    fuelsystem_7       0.065707  -0.032129  -0.008245 -0.023158  -0.066778   
    
                      curbweight  enginesize  boreratio    stroke  \
    symboling          -0.227691   -0.105790  -0.130051 -0.008735   
    wheelbase           0.776386    0.569329   0.488750  0.160959   
    carlength           0.877728    0.683360   0.606454  0.129533   
    carwidth            0.867032    0.735433   0.559150  0.182942   
    carheight           0.295572    0.067149   0.171071 -0.055307   
    curbweight          1.000000    0.850594   0.648480  0.168790   
    enginesize          0.850594    1.000000   0.583774  0.203129   
    boreratio           0.648480    0.583774   1.000000 -0.055909   
    stroke              0.168790    0.203129  -0.055909  1.000000   
    compressionratio    0.151362    0.028971   0.005197  0.186110   
    horsepower          0.750739    0.809769   0.573677  0.080940   
    peakrpm            -0.266243   -0.244660  -0.254976 -0.067964   
    citympg            -0.757414   -0.653658  -0.584532 -0.042145   
    highwaympg         -0.797465   -0.677470  -0.587012 -0.043931   
    price               0.835305    0.874145   0.553173  0.079443   
    fueltype_1         -0.217275   -0.069594  -0.054451 -0.241829   
    aspiration_1        0.324902    0.108217   0.212614  0.222982   
    doornumber_1       -0.197379   -0.020742  -0.119258  0.011082   
    carbody_1           0.098956    0.239363   0.208089  0.043215   
    carbody_2          -0.287501   -0.216805  -0.227032  0.052316   
    carbody_3           0.099425    0.088459   0.030517  0.035630   
    carbody_4           0.164075   -0.027518   0.105719 -0.095084   
    drivewheel_1       -0.666039   -0.518391  -0.583087  0.124397   
    drivewheel_2        0.669987    0.565509   0.574105 -0.022325   
    enginelocation_1    0.050468    0.196826   0.185042 -0.138455   
    enginetype_1        0.109243    0.128248   0.158136 -0.032545   
    enginetype_2        0.250124    0.016063   0.181729 -0.084688   
    enginetype_3       -0.413293   -0.363334  -0.410383  0.366084   
    enginetype_4       -0.080295   -0.016508   0.326798 -0.522808   
    enginetype_5        0.400878    0.562403   0.119509 -0.044813   
    enginetype_6       -0.039196   -0.184762   0.000127 -0.000187   
    cylindernumber_1    0.264554    0.144878  -0.007797  0.176485   
    cylindernumber_2   -0.576463   -0.631431  -0.164076 -0.111046   
    cylindernumber_3    0.405490    0.511783   0.128365  0.068388   
    cylindernumber_4   -0.143903   -0.111081  -0.108774 -0.050450   
    cylindernumber_5    0.187964    0.335555   0.054482 -0.110878   
    cylindernumber_6   -0.039196   -0.184762   0.000127 -0.000187   
    fuelsystem_1       -0.577159   -0.442562  -0.353342 -0.234866   
    fuelsystem_2       -0.040801   -0.166946   0.000110 -0.000162   
    fuelsystem_3        0.217275    0.069594   0.054451  0.241829   
    fuelsystem_4        0.034431    0.049033   0.070030  0.144263   
    fuelsystem_5        0.520220    0.483520   0.419335 -0.110280   
    fuelsystem_6       -0.002434    0.004490  -0.004213  0.251259   
    fuelsystem_7        0.024052   -0.013327   0.025977 -0.005688   
    
                      compressionratio  ...  cylindernumber_4  cylindernumber_5  \
    symboling                -0.178515  ...          0.065707         -0.047012   
    wheelbase                 0.249786  ...         -0.120709          0.037803   
    carlength                 0.158414  ...         -0.187445          0.100413   
    carwidth                  0.181129  ...         -0.183473          0.153516   
    carheight                 0.261214  ...         -0.015076         -0.170181   
    curbweight                0.151362  ...         -0.143903          0.187964   
    enginesize                0.028971  ...         -0.111081          0.335555   
    boreratio                 0.005197  ...         -0.108774          0.054482   
    stroke                    0.186110  ...         -0.050450         -0.110878   
    compressionratio          1.000000  ...         -0.011354          0.023986   
    horsepower               -0.204326  ...         -0.099600          0.280220   
    peakrpm                  -0.435741  ...         -0.003697         -0.018411   
    citympg                   0.324701  ...          0.233665         -0.131093   
    highwaympg                0.265201  ...          0.226756         -0.140150   
    price                     0.067984  ...         -0.071388          0.199634   
    fueltype_1               -0.984356  ...          0.023020          0.023020   
    aspiration_1              0.295541  ...         -0.032857         -0.032857   
    doornumber_1             -0.177888  ...          0.079143          0.079143   
    carbody_1                 0.029623  ...         -0.014109         -0.014109   
    carbody_2                -0.202650  ...          0.097231         -0.050416   
    carbody_3                 0.188286  ...         -0.065706          0.074604   
    carbody_4                 0.016315  ...         -0.026093         -0.026093   
    drivewheel_1             -0.062683  ...          0.058926         -0.083189   
    drivewheel_2              0.105185  ...         -0.053740          0.091216   
    enginelocation_1         -0.019762  ...         -0.008532         -0.008532   
    enginetype_1             -0.002519  ...         -0.004902         -0.004902   
    enginetype_2              0.219153  ...          0.280784         -0.017458   
    enginetype_3              0.027545  ...         -0.112818         -0.112818   
    enginetype_4             -0.084328  ...         -0.019672         -0.019672   
    enginetype_5             -0.086649  ...         -0.018218          0.269069   
    enginetype_6             -0.026436  ...         -0.009877         -0.009877   
    cylindernumber_1          0.173360  ...         -0.016672         -0.016672   
    cylindernumber_2         -0.012522  ...         -0.130168         -0.130168   
    cylindernumber_3         -0.065559  ...         -0.025495         -0.025495   
    cylindernumber_4         -0.011354  ...          1.000000         -0.004902   
    cylindernumber_5          0.023986  ...         -0.004902          1.000000   
    cylindernumber_6         -0.026436  ...         -0.009877         -0.009877   
    fuelsystem_1             -0.183384  ...          0.101606         -0.048245   
    fuelsystem_2             -0.022838  ...         -0.008532         -0.008532   
    fuelsystem_3              0.984356  ...         -0.023020         -0.023020   
    fuelsystem_4             -0.055528  ...         -0.004902         -0.004902   
    fuelsystem_5             -0.311035  ...         -0.064430          0.076082   
    fuelsystem_6             -0.153726  ...         -0.015003         -0.015003   
    fuelsystem_7             -0.016654  ...         -0.004902         -0.004902   
    
                      cylindernumber_6  fuelsystem_1  fuelsystem_2  fuelsystem_3  \
    symboling                 0.245950     -0.034069      0.212471     -0.194311   
    wheelbase                -0.081174     -0.396505     -0.070124      0.308346   
    carlength                -0.057877     -0.487237     -0.049998      0.212679   
    carwidth                 -0.013699     -0.522594     -0.011834      0.233880   
    carheight                -0.238720     -0.079418     -0.206225      0.284631   
    curbweight               -0.039196     -0.577159     -0.040801      0.217275   
    enginesize               -0.184762     -0.442562     -0.166946      0.069594   
    boreratio                 0.000127     -0.353342      0.000110      0.054451   
    stroke                   -0.000187     -0.234866     -0.000162      0.241829   
    compressionratio         -0.026436     -0.183384     -0.022838      0.984356   
    horsepower                0.019250     -0.541966     -0.009630     -0.163926   
    peakrpm                   0.259380     -0.095625      0.224073     -0.476883   
    citympg                  -0.183076      0.520751     -0.153487      0.255963   
    highwaympg               -0.159173      0.528009     -0.137506      0.191392   
    price                    -0.004544     -0.501374     -0.017306      0.105679   
    fueltype_1                0.046383      0.226565      0.040070     -1.000000   
    aspiration_1             -0.066203     -0.323378     -0.057191      0.401397   
    doornumber_1              0.159463     -0.020525      0.137757     -0.191491   
    carbody_1                -0.028428     -0.084946     -0.024558      0.018635   
    carbody_2                 0.195907      0.120288      0.169240     -0.202093   
    carbody_3                -0.132390     -0.060830     -0.114369      0.185623   
    carbody_4                -0.052573      0.030349     -0.045417      0.028183   
    drivewheel_1             -0.167615      0.410403     -0.144799     -0.090342   
    drivewheel_2              0.183789     -0.464056      0.158772      0.122035   
    enginelocation_1         -0.017192     -0.083975     -0.014851     -0.040070   
    enginetype_1             -0.009877     -0.048245     -0.008532     -0.023020   
    enginetype_2             -0.035176     -0.127347     -0.030388      0.268163   
    enginetype_3             -0.227314      0.217909     -0.196371      0.020584   
    enginetype_4             -0.039637      0.127119     -0.034242     -0.092384   
    enginetype_5             -0.036707     -0.179302     -0.031711     -0.085556   
    enginetype_6              1.000000     -0.097207      0.863879     -0.046383   
    cylindernumber_1         -0.033591     -0.164082     -0.029019      0.213527   
    cylindernumber_2         -0.262272      0.345607     -0.226571     -0.020184   
    cylindernumber_3         -0.051369     -0.250917     -0.044376     -0.068594   
    cylindernumber_4         -0.009877      0.101606     -0.008532     -0.023020   
    cylindernumber_5         -0.009877     -0.048245     -0.008532     -0.023020   
    cylindernumber_6          1.000000     -0.097207      0.863879     -0.046383   
    fuelsystem_1             -0.097207      1.000000     -0.083975     -0.226565   
    fuelsystem_2              0.863879     -0.083975      1.000000     -0.040070   
    fuelsystem_3             -0.046383     -0.226565     -0.040070      1.000000   
    fuelsystem_4             -0.009877     -0.048245     -0.008532     -0.023020   
    fuelsystem_5             -0.059039     -0.634114     -0.112147     -0.302574   
    fuelsystem_6             -0.030229     -0.147658     -0.026114     -0.070457   
    fuelsystem_7             -0.009877     -0.048245     -0.008532     -0.023020   
    
                      fuelsystem_4  fuelsystem_5  fuelsystem_6  fuelsystem_7  
    symboling             0.122067      0.012532      0.181939      0.065707  
    wheelbase            -0.033294      0.348891     -0.117359     -0.032129  
    carlength            -0.004831      0.511374     -0.079790     -0.008245  
    carwidth              0.012832      0.461896     -0.046399     -0.023158  
    carheight            -0.101245      0.108685     -0.278615     -0.066778  
    curbweight            0.034431      0.520220     -0.002434      0.024052  
    enginesize            0.049033      0.483520      0.004490     -0.013327  
    boreratio             0.070030      0.419335     -0.004213      0.025977  
    stroke                0.144263     -0.110280      0.251259     -0.005688  
    compressionratio     -0.055528     -0.311035     -0.153726     -0.016654  
    horsepower            0.072562      0.628372      0.117664     -0.025056  
    peakrpm              -0.018411      0.149959      0.068748     -0.018411  
    citympg              -0.066724     -0.644489     -0.123954     -0.013083  
    highwaympg           -0.068807     -0.610813     -0.106615     -0.017848  
    price                -0.002747      0.517075     -0.061475     -0.019580  
    fueltype_1            0.023020      0.302574      0.070457      0.023020  
    aspiration_1          0.149190     -0.050041      0.394703     -0.032857  
    doornumber_1          0.079143     -0.025019      0.146272      0.079143  
    carbody_1            -0.014109      0.117876     -0.043182     -0.014109  
    carbody_2             0.097231     -0.208463      0.197165      0.097231  
    carbody_3            -0.065706      0.078094     -0.105671     -0.065706  
    carbody_4            -0.026093      0.016053     -0.079860     -0.026093  
    drivewheel_1          0.058926     -0.437655      0.132020     -0.083189  
    drivewheel_2         -0.053740      0.448977     -0.115182      0.091216  
    enginelocation_1     -0.008532      0.132429     -0.026114     -0.008532  
    enginetype_1         -0.004902      0.076082     -0.015003     -0.004902  
    enginetype_2         -0.017458      0.020749     -0.053432     -0.017458  
    enginetype_3          0.043450     -0.302922      0.132984      0.043450  
    enginetype_4         -0.019672      0.004585     -0.060209     -0.019672  
    enginetype_5         -0.018218      0.282760     -0.055759     -0.018218  
    enginetype_6         -0.009877     -0.059039     -0.030229     -0.009877  
    cylindernumber_1     -0.016672      0.084981     -0.051026     -0.016672  
    cylindernumber_2      0.037659     -0.373291      0.115259      0.037659  
    cylindernumber_3     -0.025495      0.365248     -0.078030     -0.025495  
    cylindernumber_4     -0.004902     -0.064430     -0.015003     -0.004902  
    cylindernumber_5     -0.004902      0.076082     -0.015003     -0.004902  
    cylindernumber_6     -0.009877     -0.059039     -0.030229     -0.009877  
    fuelsystem_1         -0.048245     -0.634114     -0.147658     -0.048245  
    fuelsystem_2         -0.008532     -0.112147     -0.026114     -0.008532  
    fuelsystem_3         -0.023020     -0.302574     -0.070457     -0.023020  
    fuelsystem_4          1.000000     -0.064430     -0.015003     -0.004902  
    fuelsystem_5         -0.064430      1.000000     -0.197195     -0.064430  
    fuelsystem_6         -0.015003     -0.197195      1.000000     -0.015003  
    fuelsystem_7         -0.004902     -0.064430     -0.015003      1.000000  
    
    [44 rows x 44 columns]
    


```python
plt.figure(figsize=(35,20))
correlation_matrix = car.corr()  # Calculate correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1)
plt.title('Correlation Matrix')
plt.show()
```


    
![png](output_35_0.png)
    


# Data Preprocessing


```python
# Impute missing values (if any)
car.fillna(car.mean(), inplace=True)
```

## Normalize / Scale the Data


```python
scaler = StandardScaler()
car[['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg']] = scaler.fit_transform(
    car[['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg']]
)
```

## Splitting the data ( Train-Test Split )


```python
X = car.drop('price', axis=1)  # Features
y = car['price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Model 1 

## Linear Regression


```python
# Initialize and train the model
linear_model = LinearRegression()

# Fit the model on the training data
linear_model.fit(X_train, y_train)

# Make predictions
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)
```

## -- Evaluation


```python
# Calculate evaluation metrics
metrics = ['R²', 'MAE', 'MSE', 'RMSE']
train_scores = [r2_score(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred), 
                mean_squared_error(y_train, y_train_pred), np.sqrt(mean_squared_error(y_train, y_train_pred))]
test_scores = [r2_score(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred), 
               mean_squared_error(y_test, y_test_pred), np.sqrt(mean_squared_error(y_test, y_test_pred))]
# Create DataFrame for tabular display
results_df = pd.DataFrame({
    'Metric': metrics,
    'Train Score': train_scores,
    'Test Score': test_scores
})
print(results_df)
```

      Metric   Train Score    Test Score
    0     R²  9.476416e-01  8.925567e-01
    1    MAE  1.311951e+03  2.089383e+03
    2    MSE  3.122540e+06  8.482008e+06
    3   RMSE  1.767071e+03  2.912389e+03
    


```python
# Plotting the evaluation metrics
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(metrics))
ax.bar(x - 0.2, train_scores, 0.4, label='Train', color='blue')
ax.bar(x + 0.2, test_scores, 0.4, label='Test', color='orange')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Linear Regression Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()
```


    
![png](output_47_0.png)
    


## Decision Tree Regression


```python
# Initialize DecisionTreeRegressor model
dt_model = DecisionTreeRegressor()

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)
```

## -- Evaluation


```python
# Calculate evaluation metrics
metrics = ['R²', 'MAE', 'MSE', 'RMSE']
train_scores = [r2_score(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred), 
                mean_squared_error(y_train, y_train_pred), np.sqrt(mean_squared_error(y_train, y_train_pred))]
test_scores = [r2_score(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred), 
               mean_squared_error(y_test, y_test_pred), np.sqrt(mean_squared_error(y_test, y_test_pred))]

# Create DataFrame for tabular display
results_df = pd.DataFrame({
    'Metric': metrics,
    'Train Score': train_scores,
    'Test Score': test_scores
})

# Display the results table
print(results_df)
```

      Metric   Train Score    Test Score
    0     R²      0.998654  8.896985e-01
    1    MAE     64.664634  1.912711e+03
    2    MSE  80289.710366  8.707643e+06
    3   RMSE    283.354390  2.950872e+03
    


```python
# Plotting the evaluation metrics
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(metrics))
ax.bar(x - 0.2, train_scores, 0.4, label='Train', color='blue')
ax.bar(x + 0.2, test_scores, 0.4, label='Test', color='orange')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Decision Tree Regressor')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()
```


    
![png](output_52_0.png)
    


## Random Forest Regression


```python
# Initialize RandomForestRegressor model
rf_model = RandomForestRegressor()

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
```

## -- Evaluation


```python
# Calculate evaluation metrics
train_scores_rf = [r2_score(y_train, y_train_pred_rf), 
                   mean_absolute_error(y_train, y_train_pred_rf), 
                   mean_squared_error(y_train, y_train_pred_rf), 
                   np.sqrt(mean_squared_error(y_train, y_train_pred_rf))]
test_scores_rf = [r2_score(y_test, y_test_pred_rf), 
                  mean_absolute_error(y_test, y_test_pred_rf), 
                  mean_squared_error(y_test, y_test_pred_rf), 
                  np.sqrt(mean_squared_error(y_test, y_test_pred_rf))]

# Create DataFrame for tabular display
results_rf_df = pd.DataFrame({
    'Metric': metrics,
    'Train Score': train_scores_rf,
    'Test Score': test_scores_rf
})

# Display the results table
print("Random Forest Regressor Results:")
print(results_rf_df)
```

    Random Forest Regressor Results:
      Metric    Train Score    Test Score
    0     R²       0.985170  9.592418e-01
    1    MAE     593.412141  1.229665e+03
    2    MSE  884399.423900  3.217618e+06
    3   RMSE     940.425129  1.793772e+03
    


```python
# Plotting the evaluation metrics
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - 0.2, train_scores_rf, 0.4, label='Train', color='blue')
ax.bar(x + 0.2, test_scores_rf, 0.4, label='Test', color='orange')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Random Forest Regressor')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()
```


    
![png](output_57_0.png)
    


## Gradient Boosting Regression


```python
# Initialize GradientBoostingRegressor model
gb_model = GradientBoostingRegressor()

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)
```

## -- Evaluation


```python
# Calculate evaluation metrics
train_scores_gb = [r2_score(y_train, y_train_pred_gb), 
                   mean_absolute_error(y_train, y_train_pred_gb), 
                   mean_squared_error(y_train, y_train_pred_gb), 
                   np.sqrt(mean_squared_error(y_train, y_train_pred_gb))]
test_scores_gb = [r2_score(y_test, y_test_pred_gb), 
                  mean_absolute_error(y_test, y_test_pred_gb), 
                  mean_squared_error(y_test, y_test_pred_gb), 
                  np.sqrt(mean_squared_error(y_test, y_test_pred_gb))]

# Create DataFrame for tabular display
results_gb_df = pd.DataFrame({
    'Metric': metrics,
    'Train Score': train_scores_gb,
    'Test Score': test_scores_gb
})

# Display the results table
print("Gradient Boosting Regressor Results:")
print(results_gb_df)
```

    Gradient Boosting Regressor Results:
      Metric    Train Score    Test Score
    0     R²       0.992674  9.245913e-01
    1    MAE     488.730316  1.717112e+03
    2    MSE  436929.562447  5.953066e+06
    3   RMSE     661.006477  2.439891e+03
    


```python
# Plotting the evaluation metrics
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - 0.2, train_scores_gb, 0.4, label='Train', color='blue')
ax.bar(x + 0.2, test_scores_gb, 0.4, label='Test', color='orange')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Gradient Boosting Regressor')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()
```


    
![png](output_62_0.png)
    


# Hyperparameter  Tuning


```python
# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

    Best Parameters: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
    Best Score: 0.8824698567313881
    

# Model Comparison and Final Selection


```python
# Plot Actual vs Predicted Prices for Random Forest
plt.scatter(y_test, y_test_pred_rf)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest - Actual vs Predicted Prices")
plt.show()
```


    
![png](output_66_0.png)
    



```python
# Plot Residuals for Random Forest
residuals = y_test - y_test_pred_rf  # Calculate residuals
plt.scatter(y_test_pred_rf, residuals)  # Scatter plot of Predicted Prices vs Residuals
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at 0 for reference
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Random Forest - Residuals")
plt.show()
```


    
![png](output_67_0.png)
    


# Cross-Validation


```python
# Perform cross-validation
cv_scores = cross_val_score(RandomForestRegressor(), X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-validation scores:", cv_scores)
```

    Cross-validation scores: [-4843821.62845046 -2713120.09150086 -9877540.39946481 -3432621.10299219
     -9994351.9494586 ]
    

# Feature Importance


```python
# Get feature importance
feature_importance = rf_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.show()

```


    
![png](output_71_0.png)
    


# Model Evaluation on Test Data


```python
# Final Evaluation
print(f"R²: {r2_score(y_test, y_test_pred_rf)}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred_rf)}")
print(f"MSE: {mean_squared_error(y_test, y_test_pred_rf)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_rf))}")

```

    R²: 0.9592417789384023
    MAE: 1229.6652154471544
    MSE: 3217617.854504465
    RMSE: 1793.771962793617
    

# Comparison Decision Tree, Random Forest, and Gradient Boosting


```python
# Define the models' predictions
models = {
    "Decision Tree": {'train_pred': y_train_pred, 'test_pred': y_test_pred},
    "Random Forest": {'train_pred': y_train_pred_rf, 'test_pred': y_test_pred_rf},
    "Gradient Boosting": {'train_pred': y_train_pred_gb, 'test_pred': y_test_pred_gb}
}
# Initialize metrics
metrics = ['R²', 'MAE', 'MSE', 'RMSE']
results = []
# Calculate evaluation metrics for each model
for model_name, predictions in models.items():
    # Get train and test predictions
    y_train_pred = predictions['train_pred']
    y_test_pred = predictions['test_pred'] 
    # Calculate evaluation metrics for training and test data
    train_scores = [r2_score(y_train, y_train_pred), 
                    mean_absolute_error(y_train, y_train_pred), 
                    mean_squared_error(y_train, y_train_pred), 
                    np.sqrt(mean_squared_error(y_train, y_train_pred))]
    test_scores = [r2_score(y_test, y_test_pred), 
                   mean_absolute_error(y_test, y_test_pred), 
                   mean_squared_error(y_test, y_test_pred), 
                   np.sqrt(mean_squared_error(y_test, y_test_pred))]
    # Append results to the list
    results.append([model_name] + train_scores + test_scores)
# Create DataFrame for tabular display
results_df = pd.DataFrame(results, columns=['Model', 'Train R²', 'Train MAE', 'Train MSE', 'Train RMSE',
                                            'Test R²', 'Test MAE', 'Test MSE', 'Test RMSE'])
# Display the results table
print(results_df)
# Plotting the evaluation metrics
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.2  # Bar width for spacing
for idx, model_name in enumerate(models.keys()):
    # Train scores
    ax.bar(x - width + idx * width, results[idx][1:5], width, label=f'{model_name} Train', alpha=0.7)
    # Test scores
    ax.bar(x - width + idx * width + width, results[idx][5:], width, label=f'{model_name} Test', alpha=0.7)
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Regression Models')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()
```

                   Model  Train R²   Train MAE      Train MSE  Train RMSE  \
    0      Decision Tree  0.998654   64.664634   80289.710366  283.354390   
    1      Random Forest  0.985170  593.412141  884399.423900  940.425129   
    2  Gradient Boosting  0.992674  488.730316  436929.562447  661.006477   
    
        Test R²     Test MAE      Test MSE    Test RMSE  
    0  0.889699  1912.711390  8.707643e+06  2950.871598  
    1  0.959242  1229.665215  3.217618e+06  1793.771963  
    2  0.924591  1717.112402  5.953066e+06  2439.890577  
    


    
![png](output_75_1.png)
    


# MLP Classifier


```python
# Initialize MLPRegressor instead of MLPClassifier
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)

# Train the model
mlp_model.fit(X_train, y_train)

# Make predictions
y_train_pred_mlp = mlp_model.predict(X_train)
y_test_pred_mlp = mlp_model.predict(X_test)

# Calculate regression metrics
print("MLP Train MAE:", mean_absolute_error(y_train, y_train_pred_mlp))
print("MLP Test MAE:", mean_absolute_error(y_test, y_test_pred_mlp))

print("MLP Train MSE:", mean_squared_error(y_train, y_train_pred_mlp))
print("MLP Test MSE:", mean_squared_error(y_test, y_test_pred_mlp))

print("MLP Train RMSE:", mean_squared_error(y_train, y_train_pred_mlp, squared=False))
print("MLP Test RMSE:", mean_squared_error(y_test, y_test_pred_mlp, squared=False))

print("MLP Train R²:", r2_score(y_train, y_train_pred_mlp))
print("MLP Test R²:", r2_score(y_test, y_test_pred_mlp))

```

    MLP Train MAE: 5800.954912767774
    MLP Test MAE: 6210.298123928328
    MLP Train MSE: 61711063.26465425
    MLP Test MSE: 81912313.64306048
    MLP Train RMSE: 7855.638946938323
    MLP Test RMSE: 9050.542173983858
    MLP Train R²: -0.034763599243714305
    MLP Test R²: -0.03759996932418308
    

# Naive Bayes


```python
# Initialize LabelEncoder
le = LabelEncoder()

# Combine y_train and y_test to fit the LabelEncoder on all possible labels
y_combined = np.concatenate([y_train, y_test])

# Fit the encoder on the combined data
le.fit(y_combined)

# Transform both y_train and y_test
y_train_categorical = le.transform(y_train)
y_test_categorical = le.transform(y_test)

# Initialize Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train_categorical)

# Make predictions
y_train_pred_nb = nb_model.predict(X_train)
y_test_pred_nb = nb_model.predict(X_test)

# Calculate evaluation metrics
print("Naive Bayes Train Accuracy:", accuracy_score(y_train_categorical, y_train_pred_nb))
print("Naive Bayes Test Accuracy:", accuracy_score(y_test_categorical, y_test_pred_nb))

# Classification report
print("Classification Report for Naive Bayes:")
print(classification_report(y_test_categorical, y_test_pred_nb))

```

    Naive Bayes Train Accuracy: 0.9695121951219512
    Naive Bayes Test Accuracy: 0.04878048780487805
    Classification Report for Naive Bayes:
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         1
              10       1.00      1.00      1.00         1
              14       0.00      0.00      0.00         1
              15       0.00      0.00      0.00         1
              17       0.00      0.00      0.00         0
              20       0.00      0.00      0.00         1
              26       0.00      0.00      0.00         0
              28       0.00      0.00      0.00         0
              36       0.00      0.00      0.00         0
              38       0.00      0.00      0.00         1
              41       0.00      0.00      0.00         0
              42       0.00      0.00      0.00         1
              44       0.00      0.00      0.00         1
              45       0.00      0.00      0.00         1
              46       0.00      0.00      0.00         1
              48       0.00      0.00      0.00         1
              49       0.00      0.00      0.00         0
              51       0.00      0.00      0.00         0
              52       0.00      0.00      0.00         1
              54       0.00      0.00      0.00         0
              55       0.00      0.00      0.00         1
              57       0.00      0.00      0.00         1
              58       0.00      0.00      0.00         1
              60       0.00      0.00      0.00         0
              61       0.00      0.00      0.00         1
              65       0.00      0.00      0.00         0
              66       0.00      0.00      0.00         1
              72       0.00      0.00      0.00         1
              78       0.00      0.00      0.00         1
              79       0.00      0.00      0.00         1
              82       0.00      0.00      0.00         1
              84       0.00      0.00      0.00         1
              87       0.00      0.00      0.00         1
              88       0.00      0.00      0.00         0
              95       0.00      0.00      0.00         1
              99       0.00      0.00      0.00         0
             100       0.00      0.00      0.00         1
             104       0.00      0.00      0.00         0
             105       0.00      0.00      0.00         1
             110       0.00      0.00      0.00         1
             112       0.00      0.00      0.00         1
             116       0.00      0.00      0.00         1
             117       0.00      0.00      0.00         1
             119       1.00      1.00      1.00         1
             125       0.00      0.00      0.00         1
             126       0.00      0.00      0.00         0
             134       0.00      0.00      0.00         0
             139       0.00      0.00      0.00         0
             151       0.00      0.00      0.00         1
             153       0.00      0.00      0.00         0
             154       0.00      0.00      0.00         1
             165       0.00      0.00      0.00         1
             171       0.00      0.00      0.00         1
             172       0.00      0.00      0.00         1
             173       0.00      0.00      0.00         1
             174       0.00      0.00      0.00         1
             176       0.00      0.00      0.00         0
             186       0.00      0.00      0.00         1
             187       0.00      0.00      0.00         1
    
        accuracy                           0.05        41
       macro avg       0.03      0.03      0.03        41
    weighted avg       0.05      0.05      0.05        41
    
    

# XGBoost


```python
# Initialize XGBoost Regressor model
xg_model = xgb.XGBRegressor()

# Train the model
xg_model.fit(X_train, y_train)

# Make predictions
y_train_pred_xg = xg_model.predict(X_train)
y_test_pred_xg = xg_model.predict(X_test)

# Calculate evaluation metrics
print("XGBoost Train R2:", r2_score(y_train, y_train_pred_xg))
print("XGBoost Test R2:", r2_score(y_test, y_test_pred_xg))

# Calculate Mean Squared Error
print("XGBoost Train MSE:", mean_squared_error(y_train, y_train_pred_xg))
print("XGBoost Test MSE:", mean_squared_error(y_test, y_test_pred_xg))

```

    XGBoost Train R2: 0.9986532746417341
    XGBoost Test R2: 0.9341680985913954
    XGBoost Train MSE: 80315.78791987168
    XGBoost Test MSE: 5197034.999348447
    

# SCR


```python
# Initialize the SVR model
svr_model = SVR()

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions
y_train_pred_svr = svr_model.predict(X_train)
y_test_pred_svr = svr_model.predict(X_test)

# Evaluate performance
print("SVR Train MSE:", mean_squared_error(y_train, y_train_pred_svr))
print("SVR Test MSE:", mean_squared_error(y_test, y_test_pred_svr))
print("SVR Train R2:", r2_score(y_train, y_train_pred_svr))
print("SVR Test R2:", r2_score(y_test, y_test_pred_svr))

```

    SVR Train MSE: 66279908.96687816
    SVR Test MSE: 87029871.44049665
    SVR Train R2: -0.11137344799882753
    SVR Test R2: -0.10242511681999389
    

# Model Comparison and Evaluation


```python
# Create a dictionary of the models and their corresponding predictions
models = {
    'SVR': y_test_pred_svr,
    'MLP': y_test_pred_mlp,
    'Naive Bayes': y_test_pred_nb,
    'XGBoost': y_test_pred_xg
}

# Store the evaluation metrics for regression
results = []

for model_name, y_pred in models.items():
    # Calculate mean squared error and R^2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Append the results
    results.append({
        'Model': model_name,
        'Mean Squared Error': mse,
        'R-squared': r2
    })

# Convert the results into a DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

```

             Model  Mean Squared Error  R-squared
    0          SVR        8.702987e+07  -0.102425
    1          MLP        8.191231e+07  -0.037600
    2  Naive Bayes        2.578659e+08  -2.266440
    3      XGBoost        5.197035e+06   0.934168
    


```python

```