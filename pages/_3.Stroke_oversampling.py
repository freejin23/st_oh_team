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

import imblearn.over_sampling as osam 

st.set_page_config(
    page_title="Likelion AI School Oh Team Miniproject",
    page_icon="🧠",
    layout="wide",
)

st.markdown("# Stroke Prediction Dataset (kaggle) 🧠")
st.sidebar.markdown("# Stroke Oversampling🧠")

st.write("""
#### 데이터 불균형 확인(imbalanced).
""")

st.write("""
#### 1. Stroke Countplot
""")

data = pd.read_csv("content/stroke.csv")
stroke = data

bmi_group = stroke.groupby(["stroke"])[["bmi"]].mean()

stroke.loc[(stroke["stroke"] == 0) & (stroke["bmi"].isnull()), "bmi"] = bmi_group.loc[0, "bmi"]
stroke.loc[(stroke["stroke"] == 1) & (stroke["bmi"].isnull()), "bmi"] = bmi_group.loc[1, "bmi"]

num_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = stroke.drop(num_cols, axis=1)

num_cols = ["age", "avg_glucose_level", "bmi"]

stroke.loc[stroke["bmi"] > 46.29999999999999, "bmi"] = 46.29999999999999
stroke.loc[stroke["bmi"] < 10.300000000000006, "bmi"] = 10.300000000000006

stroke.loc[stroke["avg_glucose_level"] > 200, "avg_glucose_level"] = 169
stroke.loc[stroke["avg_glucose_level"] < 21, "avg_glucose_level"] = 21

stroke = stroke.drop(["id"], axis=1)

stroke = stroke.drop(3116, axis=0)

target = stroke["stroke"]
num_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]

stroke["avg_glucose_level"] = np.log1p(stroke["avg_glucose_level"])

scale = StandardScaler()
stroke[num_cols] = scale.fit_transform(stroke[num_cols])

stroke = stroke.drop("stroke", axis=1)
stroke = pd.get_dummies(stroke)

stroke = pd.concat([stroke, target], axis=1)


print(stroke["stroke"].value_counts())

fig, ax = plt.subplots()
sns.countplot(data=stroke, x="stroke", ax=ax)
st.write(fig)

st.write("""
#### 2. Stroke Oversampling
""")

X = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=25)

ros = osam.RandomOverSampler(random_state=25)
X_train, y_train = ros.fit_resample(X_train, y_train)

target = y_train.value_counts()


fig, ax = plt.subplots()
sns.barplot(x=target.index, y=target, ax=ax)
st.write(fig)

