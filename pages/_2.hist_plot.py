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
st.sidebar.markdown("# Stroke Histogram🧠")

st.write("""
#### 범주별 히스토그램.
""")

data = pd.read_csv("content/stroke.csv")
stroke = data

st.write("""
#### "bmi", "hypertension"
""")
fig, ax = plt.subplots()
sns.histplot(data=stroke, x="bmi", hue="hypertension", kde=True, ax=ax)
st.write(fig)

st.write("""
#### "bmi", "hypertension"
""")
fig, ax = plt.subplots()
sns.histplot(data=stroke, x="bmi", hue="hypertension", kde=True, ax=ax)
st.write(fig)

st.write("""
#### "bmi", "heart_disease"
""")
fig, ax = plt.subplots()
sns.histplot(data=stroke, x="bmi", hue="heart_disease", kde=True, ax=ax)
st.write(fig)

st.write("""
#### "avg_glucose_level", "smoking_status"
""")
fig, ax = plt.subplots()
sns.histplot(data=stroke, x="avg_glucose_level", hue="smoking_status", kde=True, ax=ax)
st.write(fig)

st.write("""
#### "bmi", "smoking_status"
""")
fig, ax = plt.subplots()
sns.histplot(data=stroke, x="bmi", hue="smoking_status", kde=True, ax=ax)
st.write(fig)
