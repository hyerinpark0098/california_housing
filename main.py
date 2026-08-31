import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import (fetch_california_housing,)
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 저장
df = fetch_california_housing(as_frame=True).frame
df.to_csv("california_housing.csv")

# 데이터 정보 확인
# print("df.head() : \n", df.head())
# print("df.shape : \n",df.shape)
# print("df.info() : \n",df.info())
# print("df.describe() : \n",df.describe())

# 결측치
print("columns별 결측치 갯수 확인 \n")
print(df.isna().sum())
