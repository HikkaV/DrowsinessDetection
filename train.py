import random
import numpy as np
import cv2
import keras
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from math import ceil
from skimage import transform
from skopt import forest_minimize
from tqdm import tqdm

from utils.settings import *
from f import resnet, mobnet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from utils.helpers import plot_pred
from preprocess import Process


class Train:
    def __init__(self):
        self.datagen = ImageDataGenerator(rescale=1. / 255,

                                          vertical_flip=True,
                                          fill_mode='nearest')
        self.test_df, self.valid_df, self.train_df, columns = self.read_df()
        self.classes = classes
        self.valid_data = self.datagen.flow_from_dataframe(dataframe=self.valid_df,
                                                           target_size=(dim[0], dim[1]), color_mode='rgb',
                                                           batch_size=valid_batch,
                                                           x_col='im_name',
                                                           y_col=columns,
                                                           class_mode='other',
                                                           shuffle=False,
                                                           seed=random_state,
                                                           )
        self.train_data = self.datagen.flow_from_dataframe(dataframe=self.train_df,
                                                           target_size=(dim[0], dim[1]), color_mode='rgb',
                                                           batch_size=train_batch,
                                                           x_col='im_name',
                                                           y_col=columns,
                                                           class_mode='other',
                                                           shuffle=False,
                                                           seed=random_state,
                                                           )
        self.counter = 0
        self.history_df = pd.DataFrame()
        self.preproc = Process(cfg_darnket, weights_darknet)

    def make_id(self):
        self.id_ = random.randint(random.randint(25, 601), random.randint(602, 888))

    @staticmethod
    def rescale_landmarks(df):
        overall_list = []
        for i in df.iterrows():
            overall_list.append(Train.resize_landmarks(i[1][1], i[1][2], dim[0], dim[1], i[1][3:]))
        return overall_list

    def read_df(self):

        train_df = pd.read_csv(path_to_train)
        test_df = pd.read_csv(path_to_valid)
        test_df['im_name'] = test_df['im_name'].apply(lambda x: abs_path + x)
        train_df['im_name'] = train_df['im_name'].apply(lambda x: abs_path + x)
        train_df[train_df.columns[3:]] = Train.rescale_landmarks(df=train_df)
        train_df[train_df.columns[3:]] = Train.scale(train_df)
        train_df, valid_df = train_test_split(train_df, random_state=random_state)
        columns = list(train_df.columns[3:])
        return test_df, valid_df, train_df, columns

    @staticmethod
    def scale(df):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(df[df.columns[3:]])
        Train.save_scaler(scaler)
        return arr

    @staticmethod
    def save_scaler(scaler):
        path_to_scaler = 'scaler.pickle'
        with open(path_to_scaler, 'wb') as f:
            pickle.dump(scaler, f)

    def define_params_nn(self, params):
        """
        define best params of model with generator

        """
        self.counter += 1
        self.make_id()
        eta, dropout, dropout_global, l2, epochs = params
        self.epochs = epochs
        if res_net:
            additional_str = 'resnet'
            dn = resnet.ResNet_Custom(loss=loss, dropout=dropout,
                                      eta=eta, reg=l2, weights=weights, classes=classes, dim=(dim[0], dim[1], 3))
        else:
            additional_str = 'mobnet'
            dn = mobnet.Mobnet_Custom(loss=loss, dropout=dropout,
                                      eta=eta, reg=l2, weights=weights, classes=classes, dim=(dim[0], dim[1], 3),
                                      dropout_global=dropout_global)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

        callbacks = [reduce_lr]

        self.model = dn.create_model()

        hist = self.model.fit_generator(callbacks=callbacks, generator=self.train_data,
                                        validation_data=self.valid_data,
                                        validation_steps=ceil(self.valid_data.samples / valid_batch),
                                        steps_per_epoch=ceil(self.train_data.samples / train_batch),
                                        epochs=self.epochs,
                                        )
        overall_list = []
        for i in range(epochs):
            overall_list.append((train_batch, valid_batch, dropout, dropout_global, eta, loss, self.id_,))

        df = pd.DataFrame(data=overall_list, columns=['train_batch', 'valid_batch',
                                                      'dropout', 'dropout_global', 'eta',
                                                      'loss'  'experiment_id',
                                                      'l2'])
        loss_ = hist.history['val_loss'][len(hist.history['val_loss']) - 1]
        self.model.save('models/model{}~{}~loss{}.h5'.format(self.id_, additional_str, loss_))
        for i in hist.history.keys():
            df[i] = hist.history[i]
        self.history_df = self.history_df.append(df)

        return loss_

    @staticmethod
    def write_best_params(params):
        with open('best_param.json', 'wb') as f:
            pickle.dump(params, f)

    def run_minimize(self):

        params = forest_minimize(self.define_params_nn, dimensions=space_params, n_calls=ncalls,
                                 verbose=True,
                                 random_state=random_state)

        Train.write_best_params(params)
        self.history_df.to_csv(path_to_hist, index=False)
        print('Best params are : {}'.format(params))

    def predict_normal(self, path_to_model, path_to_img, path_to_scaler='scaler.pickle', ifsave=True, name='slide.png',
                       thr=False):
        if not isinstance(path_to_model, str):
            model = path_to_model
        else:
            model = keras.models.load_model(path_to_model)

        np_image = cv2.imread(path_to_img)

        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_image_ = np.array(np_image).astype('float32') / 255
        np_image_ = transform.resize(np_image_, (1, dim[0], dim[1], 3))

        tmp = model.predict(np_image_)
        output = self.detransform(tmp, path_to_scaler).flatten()
        output = Train.resize_landmarks(dim[0],dim[1],np_image.shape[0], np_image.shape[1], output)

        if thr:
            plot_pred(np_image, output, ifsave, name)

        return list(map(int, output))

    def predict(self, path_to_model, path_to_img, path_to_scaler='scaler.pickle', ifsave=True, name='slide.png',
                thr=False):
        if not isinstance(path_to_model, str):
            model = path_to_model
        else:
            model = keras.models.load_model(path_to_model)
        np_image = cv2.imread(path_to_img)

        np_image__ = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_image, tup_ = self.preproc.crop_only_image(np_image__)
        cv2.imwrite('smth.png', np.array(np_image))
        np_image_ = np.array(np_image).astype('float32') / 255
        np_image_ = transform.resize(np_image_, (1, dim[0], dim[1], 3))

        tmp = model.predict(np_image_)
        output = self.detransform(tmp, path_to_scaler).flatten()
        output = Train.resize_landmarks(dim[0], dim[1], np_image.width, np_image.height, output)
        out = []
        x = [output[i] for i in range(len(output)) if i % 2 == 0]
        y = [output[i] for i in range(len(output)) if i % 2 != 0]
        for i in range(len(x)):
            out.append(x[i] + tup_[0])
            out.append(y[i] + tup_[1])
        output = out
        if thr:
            plot_pred(np_image__, output, ifsave, name)
        return list(map(int, output))

    @staticmethod
    def resize_landmarks(cur_width, cur_height, new_width, new_height, landmarks):
        output = []
        x = [landmarks[i] for i in range(len(landmarks)) if i % 2 == 0]
        y = [landmarks[i] for i in range(len(landmarks)) if i % 2 != 0]
        for i in range(len(x)):
            output.append(new_width / cur_width * x[i])
            output.append(new_height / cur_height * y[i])

        return output

    def detransform(self, output, path_to_scaler='scaler.pickle'):
        with open(path_to_scaler, 'rb') as f:
            scaler = pickle.load(f)
        output = scaler.inverse_transform(output)
        return output

    def make_data(self, model, df, abs_path1, thr=True):
        test_df = df
        im_names = test_df.im_name.values
        print(len(self.train_df.columns[3:]))

        overall_list = []
        for i in tqdm(im_names):
            output = self.predict_normal(model, abs_path1 + i, thr=thr)
            print(len(output))
            overall_list.append((i, *output))
            print(len(list((i, *output))))
        smth = ['im_name'] + list(self.train_df.columns[3:])
        tmp = pd.DataFrame(data=overall_list, columns=smth)
        tmp = test_df.merge(tmp, left_on='im_name', right_on='im_name', how='inner')
        tmp.to_csv('task2data.csv', index=False)
