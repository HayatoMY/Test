import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#CSVファイルを読み込む
df_train = pd.read_csv('sample/csv/train.csv')
df_test = pd.read_csv('sample/csv/test.csv')


#読み込んだCSVを確認する
#print(df_train.head())
#print(df_test.head())

#平均値、標準偏差、最小、最大値の確認
#print(df_train.describe())

#欠損している項目の確認
df_train.isnull().sum()

#欠損値のある項目を削除（行or列）
#df_train = df_train.dropna(axis=1)
#print(df_train.head())

#生存者・死亡者数（ヒストグラム）
#plt.hist(df_train['Survived'], bins=3)

#チケットの価格分布（ヒストグラム）
#plt.hist(df_train['Fare'], bins=15)

#性別分布（ヒストグラム）
#plt.hist(df_train['Sex'], bins=3)

#グラフの出力
#plt.show()

# クロス集計+割合/パラメータ表示（normalize）
#df_survived = pd.crosstab(df_train['Sex'], df_train['Survived'],normalize='columns')

# 生存率・死亡率の男女比を取得
# 生存者の男女比
#df_survived_1 = df_survived[1].values

# 死亡者の男女比
#df_survived_0 = df_survived[0].values.tolist()

# 生存率の男女比のグラフ化
#plt.bar(x=np.array(['female','male']), height=df_survived_1)
#plt.show()

x = df_train[['Pclass','SibSp','Parch']]
y = df_train['Survived']

#x、yをトレーニングデータとテストデータに分ける
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

knn = KNeighborsClassifier()

#トレーニングデータ学習させる
knn.fit(x_train,y_train)

#学習用データの予測精度を確認
print(knn.score(x_test,y_test))

#test.csvファイルでの予測
x_for_submit = df_test[['Pclass','SibSp','Parch']]
submit = df_test[['PassengerId']]
submit['Survived'] = knn.predict(x_for_submit)

#CSVファイルとして格納
submit.to_csv('sample/csv/submit01.csv' , index=False)