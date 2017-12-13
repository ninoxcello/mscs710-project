import math
import os

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle


save_dir = os.path.join(os.getcwd(), 'saved_models')

def _check_dir():
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


class MLP(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='multilayer_preceptron'):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_recent = x_recent
        self.model = None
        self.file_name = '{}.h5'.format(name)
        _check_dir()
        self.model_path = os.path.join(save_dir, self.file_name)

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(1024, input_dim=6, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=2)
        self.model.save(self.model_path)
        print('Training Complete..')


    def test(self):
        self.model = keras.models.load_model(self.model_path)
        test_score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
   

    def predict(self):
        self.model = keras.models.load_model(self.model_path)
        forecast_set = self.model.predict(self.x_recent)
        print('\nPredicted Values: {} \n'.format(forecast_set))
        return forecast_set

class LR(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='linear_regression'):
        self.clf = LinearRegression(n_jobs=-1)
        self.file_name = '{}.pickle'.format(name)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_recent = x_recent
        _check_dir()
        self.model_path = os.path.join(save_dir, self.file_name)

    def train(self):
        self.clf.fit(self.x_train, self.y_train)
        # Dump trained clf to load later
        accuracy = self.clf.score(self.x_train, self.y_train)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        print('Training Complete..')
        print('Linear Regression Model Training Accuracy: {}%'.format(accuracy * 100))

    def test(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        accuracy = self.clf.score(self.x_test, self.y_test)
        print('Linear Regression Model Accuracy: {}%'.format(accuracy * 100))

    def predict(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        forecast_set = self.clf.predict(self.x_recent)
        print('\nPredicted Values: {}\n'.format(forecast_set))
        return forecast_set


class SVR(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='support_vector_regression'):
        self.clf = svm.SVR(kernel='linear')
        self.file_name = '{}.pickle'.format(name)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_recent = x_recent
        _check_dir()
        self.model_path = os.path.join(save_dir, self.file_name)

    def train(self):
        self.clf.fit(self.x_train, self.y_train)
        accuracy = self.clf.score(self.x_train, self.y_train)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        print('Training Complete..')
        print('Support Vector Regression Model Training Accuracy: {}%'.format(accuracy * 100))
        
    def test(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        accuracy = self.clf.score(self.x_test, self.y_test)
        print('Support Vector Regression Model Accuracy: {}%'.format(accuracy * 100))

    def predict(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        forecast_set = self.clf.predict(self.x_recent)
        print('Predicted Values: {}'.format(forecast_set))
        return forecast_set

class GBR(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='gradient_boosting_regressor'):
        self.clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, loss='ls')
        self.file_name = '{}.pickle'.format(name)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_recent = x_recent
        _check_dir()
        self.model_path = os.path.join(save_dir, self.file_name)

    def train(self):
        self.clf.fit(self.x_train, self.y_train)
        accuracy = self.clf.score(self.x_train, self.y_train)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        print('Training Complete..')
        print('Gradient Boosting Regression Model Training Accuracy: {}%'.format(accuracy * 100))

    def test(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        accuracy = self.clf.score(self.x_test, self.y_test)
        print('Gradient Boosting Regression Model Accuracy: {}%'.format(accuracy * 100))

    def predict(self):
        pickle_in = open(self.model_path, 'rb')
        self.clf = pickle.load(pickle_in)
        forecast_set = self.clf.predict(self.x_recent)
        print('Predicted Values: {}'.format(forecast_set))
        return forecast_set