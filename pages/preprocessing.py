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
st.sidebar.markdown("# Stroke Missing Values🧠")

st.write("""
#### 결측치에 대해 알아보자.
""")
st.write("""
##### Stroke Dataset Missing values Heatmap .
""")

data = pd.read_csv("content/stroke.csv")
stroke = data

stroke.isnull().sum()

fig, ax = plt.subplots()
plt.figure(figsize=(12, 6))
sns.heatmap(stroke.isnull(), yticklabels = False, ax=ax)
st.write(fig)

st.write("""
#### 1. 결측치 대체
""")
st.write("""
* bmi는 평균, 중위수, 최빈값이 비슷하므로 평균 대체
""")

bmi_group = stroke.groupby(["stroke"])[["bmi"]].mean()
st.dataframe(bmi_group)


st.write("""
#### 2. 중복값 확인
""")
st.write("""
* 중복값이 존재하지 않는다.
""")

st.write("""
#### 3. 이상치 확인
""")
st.write("""
* "age", "avg_glucose_level", "bmi"
""")
