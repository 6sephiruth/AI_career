import pandas as pd

import time
import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import numpy as np

good_data = pd.read_csv("./dataset/good.csv")
bad_data = pd.read_csv("./dataset/bad.csv")

good_data = good_data.astype(int)
bad_data = bad_data.astype(int)

if os.path.exists(f'./dataset/fix_good_datas') and os.path.exists(f'./dataset/fix_bad_data') and os.path.exists(f'./dataset/drop_cols'):
    fix_good_data = pickle.load(open(f'./fix_good_data','rb'))
    fix_bad_data = pickle.load(open(f'./fix_bad_data','rb'))
    drop_cols = pickle.load(open(f'./drop_cols','rb'))
    
else:

    drop_cols = ['Unnamed: 0', '.', '+', '-',  '/', ',', '및', '은', '을', '의', '첫', '하기', '를', '3', '된', '매', 
                 '10', '5', '에', '에서', '완료', '!', '과', '있습니다', '만', '1', '8', '50', '는', '가', '고', '가장',
                  '=', '2', '중','한', '하고', '4', '위', ')', '(', '100','30', '?', '하여', 'ㄴ', '것', '입니다', '많은',
                  '합니다', '6', '시', '모든', '다', '와', '곳', '하는', '도', '적', '인', '할', '즈', '_', '경우',
                  '우리', '일', '더', '9', '7', '0', '수', '들', ':', ']', '[', '으로', '없는', '&', '등', '로', '있는',
                   '이', '충', '|', '\'', '\'', '\"', '경기', '저희', '호텔', '365']
    for col in bad_data.columns:

        vacuum_check = col.strip()

        if bad_data[col].sum() < 5000 or len(vacuum_check) == 0:
            drop_cols.append(col)

    fix_good_data = good_data.drop(['Unnamed: 0.1'], axis=1)
    fix_good_data = fix_good_data.drop(drop_cols, 1)
    fix_bad_data = bad_data.drop(drop_cols, 1)

    fix_good_data['label'] = 0
    fix_bad_data['label'] = 1

    pickle.dump(fix_good_data, open(f'./dataset/fix_good_data','wb'))
    pickle.dump(fix_bad_data, open(f'./dataset/fix_bad_data','wb'))
    pickle.dump(drop_cols, open(f'./dataset/drop_cols','wb'))

# Shuffle
fix_good_data.sample(frac=1)
fix_bad_data.sample(frac=1)

# 데이터셋 나누기 학습용 90% 테스트용 10%
good_train = fix_good_data.sample(frac=0.9)
good_test = fix_good_data.drop(good_train.index)

bad_train = fix_bad_data.sample(frac=0.9)
bad_test = fix_bad_data.drop(bad_train.index)

train = pd.concat([good_train, bad_train], axis=0)
test = pd.concat([good_test, bad_test], axis=0)

x_train = train.iloc[:, :-1].to_numpy()
y_train = train.iloc[:, -1:].to_numpy()

x_test = test.iloc[:, :-1].to_numpy()

y_test = test.iloc[:, -1:].to_numpy()

checkpoint_path = 'dataset/cp'

if os.path.exists(f'./dataset/cp/saved_model.pb'):

    model = tf.keras.models.load_model(checkpoint_path)
    
else:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    checkpoint = ModelCheckpoint(checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)


    model = Sequential([

    Dense(128, activation='relu' ,input_shape=[14]),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test) , shuffle=True, epochs=300, verbose=1, callbacks=[checkpoint],)
    
    model.save(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_path)

model.trainable = False

print(model.evaluate(x_test, y_test)[1])