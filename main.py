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
st.sidebar.markdown("# Stroke 🧠")

st.write("""
#### Oh Team -박지현, 박준혁, 선은지, 성찬영, 이주승
""")

data = pd.read_csv("content/stroke.csv")

if st.checkbox('Show raw data'):
    data_load_state = st.text('Loading data...')
    st.subheader('Stroke Prediction Dataset (kaggle)')
    stroke = data
    st.dataframe(stroke.style.highlight_max(axis=0))
    data_load_state.text("Done!")
    
stroke.shape
stroke.info()


st.write("""
#### 1. 연속형 변수
""")
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


# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)
