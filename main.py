import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import (fetch_california_housing,)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# 결측치 확인
# print("columns별 결측치 갯수 확인 \n")
# print(df.isna().sum())

# 데이터 탐색
# 단변량
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
# sns.histplot(df["MedInc"], bins=20, ax=axes[0,0])
# axes[0,0].set_title("해당 지역의 중위소득")
# sns.histplot(df["HouseAge"], bins=20, ax=axes[0,1])
# axes[0,1].set_title("해당 지역 주택의 연식")
# sns.histplot(df["AveRooms"], bins=20, ax=axes[0,2])
# axes[0,2].set_title("가구당 평균 방 갯수")
# sns.histplot(df["AveBedrms"], bins=20, ax=axes[1,0])
# axes[1,0].set_title("가구당 평균 침실 갯수")
# sns.histplot(df["Population"], bins=20, ax=axes[1,1])
# axes[1,1].set_title("해당 지역의 인구수")
# sns.histplot(df["AveOccup"], bins=20, ax=axes[1,2])
# axes[1,2].set_title("가구당 평균 거주인원")
# sns.histplot(df["Latitude"], bins=20, ax=axes[2,0])
# axes[2,0].set_title("위도")
# sns.histplot(df["Longitude"], bins=20, ax=axes[2,1])
# axes[2,1].set_title("경도")
# sns.histplot(df["MedHouseVal"], bins=20, ax=axes[2,2])
# axes[2,2].set_title("주택 가격의 중앙값")

# 이변량
# fig, axes = plt.subplots(2,4, figsize=(15,12))
# sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", ax=axes[0,0])
# axes[0,0].set_title("중위소득과 주택가격의 상관관계")
#
# sns.scatterplot(data=df, x="HouseAge", y="MedHouseVal", ax=axes[0,1])
# axes[0,1].set_title("주택연식과 주택가격의 상관관계")
#
# sns.scatterplot(data=df, x="AveRooms", y="MedHouseVal", ax=axes[0,2])
# axes[0,2].set_title("방 갯수와 주택가격의 상관관계")
# axes[0,2].set_xlim(0,5)
#
# sns.scatterplot(data=df, x="AveBedrms", y="MedHouseVal", ax=axes[0,3])
# axes[0,3].set_title("침실 갯수와 주택가격의 상관관계")
# axes[0,3].set_xlim(0,5)
#
# sns.scatterplot(data=df, x="Population", y="MedHouseVal", ax=axes[1,0])
# axes[1,0].set_title("인구수와 주택가격의 상관관계")
#
# sns.scatterplot(data=df, x="AveOccup", y="MedHouseVal", ax=axes[1,1])
# axes[1,1].set_title("가구당 평균 거주인원과 주택가격의 상관관계")
# axes[1,1].set_xlim(0,5)
#
# sns.scatterplot(data=df, x="Latitude", y="MedHouseVal", ax=axes[1,2])
# axes[1,2].set_title("위도와 주택가격의 상관관계")
#
# sns.scatterplot(data=df, x="Longitude", y="MedHouseVal", ax=axes[1,3])
# axes[1,3].set_title("경도와 주택가격의 상관관계")
#
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 8))
#
# corr = df.corr(numeric_only=True)
#
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#
# plt.title("칼럼별 상관관계")
# plt.tight_layout()
# plt.show()

# 데이터 학습 준비
features = ["MedInc", "AveRooms", "Latitude", "Longitude", "HouseAge"]

X = df[features]
y = df["MedHouseVal"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 학습 예측
m = LinearRegression()
m.fit(X_train, y_train)

y_pred = m.predict(X_test)
print(y_pred)

# 평가
#r2
r2 = r2_score(y_test, y_pred)
print("r2:", r2)

mae = mean_absolute_error(y_test, y_pred)
print("mae:", mae)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse:", rmse)