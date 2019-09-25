import pickle
from math import ceil
import  numpy as np
from keras import backend as K, Input
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from train import Train
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from nets_custom import resnet

tr = Train()
abs_path = '/home/hikkav/hack/data/eye_tracking/train_images/'

# model = keras.models.load_model('models/model819~resnet~loss0.0715680956130936.h5')


def create_model(dim):
    # Input(shape=)
    K.concatenate([Dense(2000,activation='relu')])
    model.add(Dense(2000, input_dim=dim, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(1500, activation="relu", kernel_regularizer=l2(0.1)))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation="relu", kernel_regularizer=l2(0.1)))
    # model.add(BatchNormalization())
    # model.add(Dense(500, activation="relu", kernel_regularizer=l2(0.1)))

    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Dense(2, activation="sigmoid"))

    # return our model
    return model


def load_vectors():
    df = pd.read_csv('/home/hikkav/hack/data/eye_tracking/train.csv')
    return df


def make_overall():
    tr = Train()
    df = load_vectors()
    tr.make_data(model,
                 df, )


def save_scaler(scaler, path):
    path_to_scaler = path
    with open(path_to_scaler, 'wb') as f:
        pickle.dump(scaler, f)


def fit_regression(dataset_path):
    df = pd.read_csv(dataset_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(df[df.columns[3:]])
    save_scaler(scaler, 'scalertask2_1.pickle')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Y = scaler.fit_transform(df[['x', 'y']])
    save_scaler(scaler, 'scalertask2_2.pickle')
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model = create_model(34)
    opt = Adam(lr=1e-4, decay=1e-3 / 20)
    model.compile(loss="mean_absolute_error", optimizer=opt)
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=20, verbose=1, steps_per_epoch=ceil(X_train.shape[0] / 64),
                     validation_steps=ceil(X_test.shape[0] / 64))

    return model


def detransform(output, path_to_scaler='scaler.pickle'):
    with open(path_to_scaler, 'rb') as f:
        scaler = pickle.load(f)
    output = scaler.inverse_transform(output)
    return output

def transform(output, path_to_scaler='scaler.pickle'):
    with open(path_to_scaler, 'rb') as f:
        scaler = pickle.load(f)
    output = scaler.transform(output)
    return output

def sumbit(test_name, output_filename, model2):
    tr = Train()
    df = pd.read_csv(test_name)
    im_names = df['im_name']
    prediction_x = []
    prediction_y = []
    im_name = []
    for im_name_ in im_names:
        print(im_name_)
        first_model_output = tr.predict_normal(model, '/home/hikkav/hack/data/eye_tracking/test_images/'+im_name_, thr=True)
        print(first_model_output)
        X = transform(np.array(first_model_output).reshape(1,-1),'scalertask2_1.pickle')
        pred = model2.predict(X)
        pred = detransform(pred, 'scalertask2_2.pickle')
        pred = np.array(list(map(int,pred[0])))
        print(pred)
        prediction_x.append(pred[0])
        prediction_y.append(pred[1])


    df['im_name'] = im_names
    df['x'] = prediction_x
    df['y'] = prediction_y
    df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    # model = fit_regression('task2data.csv')
    # model.save('regression.h5')
    # model2 = keras.models.load_model('regression.h5')

    # sumbit('/home/hikkav/hack/data/eye_tracking/test.csv', 'sub2.csv', model2)
    make_overall()