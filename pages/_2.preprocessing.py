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
st.sidebar.markdown("# Stroke Preprocessing🧠")

st.write("""
#### 데이터 전처리 과정.
""")

st.write("""
#### 1. 결측치 확인
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
* bmi는 평균, 중위수, 최빈값이 비슷하므로 평균 대체
""")

bmi_group = stroke.groupby(["stroke"])[["bmi"]].mean()
st.dataframe(bmi_group)

stroke.loc[(stroke["stroke"] == 0) & (stroke["bmi"].isnull()), "bmi"] = bmi_group.loc[0, "bmi"]
stroke.loc[(stroke["stroke"] == 1) & (stroke["bmi"].isnull()), "bmi"] = bmi_group.loc[1, "bmi"]


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

def outlier_df(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    outlier = data[(Q1 - 1.5 * IQR > data[col]) | (data[col] > Q3 + 1.5 * IQR)] 
    return outlier

body1 = """
outlier_df(stroke, "avg_glucose_level")
하한값 : 21.977500000000006,  상한값 : 169.35750000000002
"""
st.code(body1, language="python")

st.dataframe(outlier_df(stroke, "avg_glucose_level"))

body1 = """
outlier_df(stroke, "bmi")
하한값 : 10.300000000000006,  상한값 : 46.29999999999999
"""
st.code(body1, language="python")

st.dataframe(outlier_df(stroke, "bmi"))

body1 = """
outlier_df(stroke, "bmi")
하한값 : 10.300000000000006,  상한값 : 46.29999999999999
"""
st.code(body1, language="python")

st.dataframe(outlier_df(stroke, "bmi"))

st.write("""
##### boxplot .
""")
num_cols = ["age", "avg_glucose_level", "bmi"]

fig, ax = plt.subplots()
plt.figure(figsize=(12, 6))
sns.boxplot(data= stroke[num_cols], ax=ax)
st.write(fig)

st.write("""
* bmi 이상치를 각 상한값과 하한값으로 대체
""")

stroke.loc[stroke["bmi"] > 46.29999999999999, "bmi"] = 46.29999999999999
stroke.loc[stroke["bmi"] < 10.300000000000006, "bmi"] = 10.300000000000006

st.write("""
* 혈당 포도당 수치가 공복, 식전, 식후 중 어느 것을 기준으로 할지 혼란.
* 200이 넘는 값을 상한값으로 대체한다.
""")

stroke.loc[stroke["avg_glucose_level"] > 200, "avg_glucose_level"] = 169
stroke.loc[stroke["avg_glucose_level"] < 21, "avg_glucose_level"] = 21

st.write("""
#### 4. 불필요한 id 컬럼 삭제
""")
stroke = stroke.drop(["id"], axis=1)

st.write("""
#### 5. gender 컬럼에서 불필요한 Other 성별 삭제
""")
stroke[stroke["gender"] == "Other"]
stroke = stroke.drop(3116, axis=0)

st.write("""
#### 6. 스케일링
""")

st.write("""
#### 6-1. avg_glucose_level log 변환
""")
target = stroke["stroke"]
num_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]

st.write("""
* avg_glucose_level log 변환 전
""")
stroke["avg_glucose_level"].hist(figsize=(16, 10), bins=50)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.write("""
* avg_glucose_level log 변환 후
""")
np.log1p(stroke["avg_glucose_level"]).hist(figsize=(16, 10), bins=50)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

stroke["avg_glucose_level"] = np.log1p(stroke["avg_glucose_level"])

st.write("""
#### 6-2. 연속형 변수 StandardScaler
""")

scale = StandardScaler()
stroke[num_cols] = scale.fit_transform(stroke[num_cols])

st.write("""
#### 6-3. 범주형 변수 One-hot Endoding
""")

stroke = stroke.drop("stroke", axis=1)
stroke = pd.get_dummies(stroke)

stroke = pd.concat([stroke, target], axis=1)

st.write("""
#### 전처리 후 데이터
""")

st.dataframe(stroke)
