import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import kstest

from sklearn.svm import SVC

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree
from lightgbm import LGBMClassifier
from lightgbm import plot_importance, plot_metric, plot_tree

from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import power_transform
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle



st.set_page_config(
    page_title="Likelion AI School Oh Team Miniproject",
    page_icon="🧠",
    layout="wide",
)

st.markdown("# Stroke Prediction Dataset (kaggle) 🧠")
st.sidebar.markdown("# Stroke Data Info🧠")

st.write("""
#### Oh Team -박지현, 박준혁, 선은지, 성찬영, 이주승
""")

data = pd.read_csv("content/stroke.csv")
stroke = data

if st.checkbox('Show raw data'):
    data_load_state = st.text('Loading data...')
    st.subheader('Stroke Prediction Dataset (kaggle)')
    df = data
    st.dataframe(df.style.highlight_max(axis=0))
    data_load_state.text("Done!")

st.write("""
#### Data Description
""")

body = """
- `id` : 고유 식별자
- `gender` : 환자의 성별
- `age` : 환자의 연령
- `hypertension` : 환자가 고혈압이 아닌 경우 0, 고혈압인 경우 1
- `heart_parament` : 환자가 심장 질환이 없는 경우 0, 환자가 심장 질환이 있는 경우 1
- `ever_married` : 결혼한 적이 있는가 "아니오" 또는 "예"
- `work_type` : "Children", "Govt_jov", "Never_worked", "Private" 또는 "Self-employed"
- `Residence_type` : "Rural" or "Urban”
- `avg_glucose_level` : 혈중 평균 포도당 수준
- `bmi` : 체질량지수
- `smoking_status` : "formerly smoked", "never smoked", "smokes" or "Unknown"
- `stroke` : 뇌졸중이 있는 경우 1, 뇌졸중이 아닌 경우 0
- 참고 : smoking_status의 **Unknown**은 이 환자에 대한 정보를 사용할 수 없음을 의미한다.
"""
st.code(body, language="python")


st.write("""
#### 1. 데이터 크기
""")
st.write("""
* (5110, 12)
""")
st.write("""
#### 2. 데이터 정보
""")
body = """
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
"""
st.code(body, language="python")

st.write("""
#### 3. 연속형 변수
""")

body1 = """
"age", "bmi", "avg_glucose_level"
"""
st.code(body1, language="python")

sample1 = stroke[["age", "bmi", "avg_glucose_level"]]

def num_desc(data):
    df = data.describe().T
    
    df1 = data.isnull().sum()
    df1.name = "missing"
    
    df2 = data.skew()
    df2.name = "skewness"
    
    df3 = data.kurt()
    df3.name = "kurtosis"

    df4 = data.median()
    df4.name = "median"
    
    df = pd.concat([df, df1, df2, df3, df4], axis=1)
    df["total"] = df["count"] + df["missing"]
    
    order = ["total", "count", "missing", "mean", "median", "std", "skewness", "kurtosis", "min", "max", "25%", "75%"]
        
    num_df = df[order]
    num_df = num_df.round(2)
    
    return num_df

st.dataframe(num_desc(sample1))

st.write("""
#### 4. 범주형 변수
""")

def cat_df(data, col):
    cat_df = data[col].value_counts(dropna=False).to_frame().sort_index(ascending=True).rename(columns={col:"count"}).reset_index()
    cat_df = cat_df.rename(columns={"index":col})

    return cat_df

body2 = """
"gender"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "gender"))

body2 = """
"ever_married"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "ever_married"))

body2 = """
"work_type"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "work_type"))

body2 = """
"Residence_type"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "Residence_type"))

body2 = """
"hypertension"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "hypertension"))

body2 = """
"heart_disease"
"""
st.code(body2, language="python")
st.dataframe(cat_df(stroke, "heart_disease"))

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)
