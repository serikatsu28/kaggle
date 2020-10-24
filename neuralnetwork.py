#1.　データの取得と欠損値の確認
import pandas as pd
import numpy as np
import os, random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

sc = StandardScaler()

#乱数を固定する関数
def reset_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed) # random関数のシードを固定
    np.random.seed(seed) # numpyのシードを固定
    tf.random.set_seed(seed) # tensorflowのシードを固定

#乱数を固定
reset_seed(28)

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#trainデータとtestデータを１つにまとめる
data = pd.concat([train,test]).reset_index(drop=True)
#欠損値が含まれる行数を確認
train.isnull().sum()
test.isnull().sum()


#2. 欠損値の補完と特徴量の作成

#2.1 Fareの補完
data['Fare'] = data['Fare'].fillna(data.query('Pclass==3 & Embarked=="S"')['Fare'].median())

#2.2 グループごとの生死の違い'Family_survival'の作成
#名前の名字を取得して'Last_name'に入れる
data['Last_name'] = data['Name'].apply(lambda x: x.split(",")[0])

data['Family_survival'] = 0.5 #デフォルトの値
#Last_nameとFareでグルーピング
for grp, grp_df in data.groupby(['Last_name', 'Fare']):

    if (len(grp_df) != 1):
        #(名字が同じ)かつ(Fareが同じ)人が2人以上いる場合
        for index, row in grp_df.iterrows():
            smax = grp_df.drop(index)['Survived'].max()
            smin = grp_df.drop(index)['Survived'].min()
            passID = row['PassengerId']

            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_survival'] = 0
            #グループ内の自身以外のメンバーについて
            #1人でも生存している → 1
            #生存者がいない(NaNも含む) → 0
            #全員NaN → 0.5

#チケット番号でグルーピング
for grp, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        #チケット番号が同じ人が2人以上いる場合
        #グループ内で1人でも生存者がいれば'Family_survival'を1にする
        for ind, row in grp_df.iterrows():
            if (row['Family_survival'] == 0) | (row['Family_survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_survival'] = 1
                elif (smin == 0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_survival'] =

#2.3 家族の人数を表す特徴量 'Family_size' の作成と階級分け
#Family_sizeの作成
data['Family_size'] = data['SibSp']+data['Parch']+1
#1, 2~4, 5~の3つに分ける
data['Family_size_bin'] = 0
data.loc[(data['Family_size']>=2) & (data['Family_size']<=4),'Family_size_bin'] = 1
data.loc[(data['Family_size']>=5) & (data['Family_size']<=7),'Family_size_bin'] = 2
data.loc[(data['Family_size']>=8),'Family_size_bin'] = 3

#2.4 名前の敬称 'Title' の作成
#名前の敬称を取得して'Title'に入れる
data['Title'] = data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
#数の少ない敬称を統合
data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
data['Title'].replace(['Mlle'], 'Miss', inplace=True)
data['Title'].replace(['Jonkheer'], 'Master', inplace=True)

#2.5 Ageの補完と階級分け
#敬称ごとの平均値でAgeの欠損値を補完
title_list = data['Title'].unique().tolist()
for t in title_list:
    index = data[data['Title']==t].index.values.tolist()
    age = data.iloc[index]['Age'].mean()
    age = np.round(age,1)
    data.iloc[index,5] = data.iloc[index,5].fillna(age)

#年齢ごとに階級分け
data['Age_bin'] = 0
data.loc[(data['Age']>18) & (data['Age']<=60),'Age_bin'] = 1
data.loc[(data['Age']>60),'Age_bin'] = 2

#2.6 Fareの標準化&特徴量のダミー変数化
#Fareを標準化したものを'Fare_std'に入れる
data['Fare_std'] = sc.fit_transform(data[['Fare']])
#ダミー変数に変換
data['Sex'] = data['Sex'].map({'male':0, 'female':1})
data = pd.get_dummies(data=data, columns=['Title','Pclass','Family_survival'])

#いらない特徴量を落とす
data = data.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket',
                     'Fare','Cabin','Embarked','Family_size','Last_name'], axis=1)

#統合させていたデータをtrainデータとtestデータに分ける
model_train = data[:891]
model_test = data[891:]

x_train = model_train.drop('Survived', axis=1)
y_train = pd.DataFrame(model_train['Survived'])
x_test = model_test.drop('Survived', axis=1)


#3. モデルの構築と予測
#モデルの初期化
model = Sequential()
#層の構築
model.add(Dense(12, activation='relu', input_dim=16))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#モデルの構築
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='acc')
#モデルの構造を表示
model.summary()
#モデルの学習
log = model.fit(x_train, y_train, epochs=5000, batch_size=32,verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=100,verbose=1)],
                validation_split=0.3)

#学習が進行する様子をグラフで表示
plt.plot(log.history['loss'],label='loss')
plt.plot(log.history['val_loss'],label='val_loss')
plt.legend(frameon=False)
plt.xlabel('epochs')
plt.ylabel('crossentropy')
plt.show()

#0と1どちらに分類されるかを予測
y_pred_cls = model.predict_classes(x_test)
#kaggleに出すデータフレームを作成
y_pred_cls = y_pred_cls.reshape(-1)
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':y_pred_cls})
submission.to_csv('titanic_nn.csv', index=False)
