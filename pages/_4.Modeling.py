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
st.sidebar.markdown("# Stroke Modeling🧠")

st.write("""
#### Vote_model.
""")

st.write("""
#### 1. model
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

X = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=25)

ros = osam.RandomOverSampler(random_state=25)
X_train, y_train = ros.fit_resample(X_train, y_train)

target = y_train.value_counts()

df = pd.concat([X_train, y_train], axis=1)

target_0 = df[df['stroke'] == 0] 
target_1 = df[df['stroke'] == 1] 

target_0 = pd.DataFrame(target_0).reset_index(drop=True)
target_1 = pd.DataFrame(target_1).reset_index(drop=True)

target_1 = target_1.sample(1701, random_state=725)

df = pd.concat([target_0, target_1], axis=0).reset_index(drop=True)


body1 = """
X_train = df.drop(["stroke"], axis=1)
y_train = df["stroke"]

model1 = LGBMClassifier(max_depth=3, n_estimators=123,random_state=25)
model2 = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=3, random_state=25)
model3 = GradientBoostingClassifier(max_depth=3, n_estimators=131, random_state=25)

vote_model = VotingClassifier(estimators=[("LGBM", model1), ("tree", model2), ("Gradient", model3)], voting="hard")

vote_model.fit(X_train, y_train)
predict = vote_model.predict(X_test)

print(vote_model.score(X_train, y_train))
print(vote_model.score(X_test, y_test))

0.8973153047227121
0.8851924331376386

"""
st.code(body1, language="python")



st.write("""
#### 2. report
""")
body1 = """
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

              precision    recall  f1-score   support

           0       0.97      0.90      0.94      1458
           1       0.21      0.51      0.30        75

    accuracy                           0.89      1533
   macro avg       0.59      0.71      0.62      1533
weighted avg       0.94      0.89      0.91      1533

"""
st.code(body1, language="python")

st.write("""
#### 3. confusion_matrix
""")

uploaded_file = st.file_uploader("confusion_matrix", type="jpg")
image = Image.open(uploaded_file)
st.image(image, caption='Uploaded Image.', use_column_width=True)


st.write("""
#### 4. final
""")
body1 = """

[[1319  139]
 [  37   38]]
정확도 : 0.89
오분류율 : 0.11
AUC : 0.71
정밀도 : 0.21
재현율 : 0.51
F1 Score : 0.3

"""
st.code(body1, language="python")
