import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn.linear_model import LinearRegression
import pickle
import math
import os


save_dir = os.path.join(os.getcwd(), 'saved_models')

def _check_dir():
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

class MLP(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='multilayerpreceptron'):
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
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=2)

        self.model.save(self.model_path)

        print('Training Complete..')
        train_score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))

    def test(self):
        self.model = keras.models.load_model(self.model_path)
        test_score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score, math.sqrt(test_score)))

    def predict(self):
        self.model = keras.models.load_model(self.model_path)
        forecast_set = self.model.predict(self.x_recent)
        print('\nPredicted Values: {} \n'.format(forecast_set))
        return forecast_set

class LR(object):
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='linearregression'):
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
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        print('Training Complete..')

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
    def __init__(self, x_train, y_train, x_test, y_test, x_recent, name='supportvectorregression'):
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
        # Dump trained clf to load later
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        print('Training Complete..')

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
