import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#CSVファイルを読み込む
df_train = pd.read_csv('sample/csv/train.csv')
df_test = pd.read_csv('sample/csv/test.csv')

#欠損データを確認
confirm_train = df_train.isnull().sum()

'''Age（177）、Cabin（687）、Embarked(2)に欠損'''

#Cabinを削除
#df_train = df_train.drop('Cabin',axis=1)

#Ageの中央値を出す　

median_age = np.nanmedian(df_train['Age'])
#NaNを除いた中央値「28.0」

'''Ageを予測して出す'''

#算出した中央値をAgeの欠損部（NaN）に代入
df_train['Age'] = df_train['Age'].fillna(median_age)

#性別を数値（0,1,2,3）に変更　⇨ map関数を使う　

df_train['Sex'] = df_train['Sex'].map({'male':0 , 'female':1})
df_test['Sex'] = df_test['Sex'].map({'male':0 , 'female':1})

# 名前（Name）から敬称（Title）を抽出
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

# 敬称をカテゴリごとに数値に変換
title_map = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
    'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 7,
    'Mlle': 2, 'Countess': 3, 'Ms': 2, 'Lady': 3,
    'Jonkheer': 7, 'Don': 7, 'Sir': 7, 'Mme': 3,
    'Capt': 7, 'Dona': 3
}

df_train['Title'] = df_train['Title'].map(title_map).fillna(0)
df_test['Title'] = df_test['Title'].map(title_map).fillna(0)

#Embarkedの欠損部（2件）を補完する
#Pclassが同じ出港地ごとの料金を比較する（中央値）

C = df_train[(df_train['Embarked'] == 'C') & (df_train['Pclass']== 1)]['Fare'].median()
#print("Cの中央値は",C)
S = df_train[(df_train['Embarked'] == 'S') & (df_train['Pclass']== 1)]['Fare'].median()
#print("Sの中央値は",S)
Q = df_train[(df_train['Embarked'] == 'Q') & (df_train['Pclass']== 1)]['Fare'].median()
#print("Qの中央値は",Q)

#Embarkedの欠損部分に部分に'C'を代入
df_train['Embarked'] = df_train['Embarked'].fillna("C")
#print(df_train['Embarked'].iloc[61])


#Name（苗字部分）、Fare、（Embarked、）SibSp、Parchから夫婦、兄弟を判別する
#一緒に来てる人をグループ（Family_Group）化する

# 姓を抽出
df_train['LastName'] = df_train['Name'].str.extract(r'([^,]+),')

#SibSpとParchを足した合計数を抽出
df_train['Family_Number_Sum'] = df_train['SibSp'] + df_train['Parch']

# グループ化キーを作成（姓 + 運賃 + 乗船場所 + チケット番号）
df_train['GroupKey'] = df_train['LastName'] + "_" + df_train['Family_Number_Sum'].astype(str) 

# グループIDを付与
df_train['GroupID'] = df_train.groupby('GroupKey').ngroup()

#チケット番号(Ticket)を確認する
#CabinをPclass、GroupID、Fare、Embarkedから予測する
'''一旦Cabinは使わずにいく'''

#FareとCabinのグラフを抽出


#Pclass、Sex、Name（敬称）、Age、SibSp、Parch、Family＿Number＿Sumと生存者の相関を見る

# 必要な列のみをデータフレーム化
df_grouped = df_train[['Survived','Title', 'Pclass', 'Fare','Sex', 'Age', 'SibSp', 'Parch', 'Family_Number_Sum','GroupID']]

# 結果をCSVに保存
df_grouped.to_csv("check_correlation_2.csv", index=False)

#負の相関が出ているカラムは除外する
#使うデータを決定する
#RandomForestモデルで予測する
