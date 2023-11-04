from ast import expr_context
from email.mime import base
import shutil
import tensorflow as tf
import uuid
import os
import config
import pandas as pd
from tensorflow import keras
from keras import activations, optimizers, layers, Sequential
from keras.models import load_model
from database import Db
from sklearn.model_selection import train_test_split

class ANN:
    
    def __init__(self, optimizer, batch_size, num_of_epochs, activation, output_activation, dataset_name): # dataset_name = 'filename.csv'
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._num_of_epochs = num_of_epochs
        self._activation = activation
        self._model_name = dataset_name.removesuffix(".csv")
        self._output_activation = output_activation
        self._dataset_name = dataset_name
        self._neurons_per_hidden_layer = [2]

        self._init_default_hyperparameters()

    def _create_model(self, num_of_inputs, num_of_outputs):
        self.model = Sequential()
        
        self.model.add(layers.Input(shape = (num_of_inputs, )))
        for unit in self._neurons_per_hidden_layer:
            self.model.add(layers.Dense(units = int(unit), activation = self._activation))
        self.model.add(layers.Dense(units = num_of_outputs, activation = self._output_activation))

        self.model.compile(optimizer = self._optimizer, loss = 'mse', metrics = ['mae'])

    def train_model(self, data, num_of_outputs):
        
        """ -------------- SPLIT DATA -------------- """
        data = pd.read_csv(data)
        data = data.sample(frac = 1)

        X, y = data.values[:, :-num_of_outputs], data.values[:, -num_of_outputs]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)
        
        self._create_model(len(data.columns) - num_of_outputs, num_of_outputs)

        self.model.fit(X_train, y_train, batch_size = self._batch_size, epochs = self._num_of_epochs, verbose = 0)

        metrics = self._evaluate_model(X_test, y_test)

        """ -------------------- UPLOAD TO S3 """
        self.save_model()
        where_to_put_zip_file = os.path.join(config.models_path, self._model_name)
        
        shutil.make_archive(where_to_put_zip_file, 'zip', os.path.join(config.models_path, self._model_name))
        Db.upload_to_s3(aws_path = self._model_name +".zip", local_file_source = where_to_put_zip_file +".zip")

        """ -------------------- SAVE TO DYNAMODB """
        Db.add_model_to_dynamoDB(dataset_name = self._dataset_name, mse = metrics['mse'], mae = metrics['mae'])

        print("mse:", metrics["mse"])
        print("mae:", metrics["mae"])

        return metrics

    def predict(model, dataset, num_of_outputs):

        X_test = pd.read_csv(dataset)
        X_test = X_test.values[:, :-num_of_outputs]
        return model.predict(X_test)

    def _evaluate_model(self, X_val, y_true):
        mse, mae = self.model.evaluate(X_val, y_true, verbose = 0)
        return {
            "mse": mse,
            "mae": mae
        }

    def save_model(self):
        base_dir = os.path.join(config.models_path, self._model_name)
        
        try:
            shutil.rmtree(base_dir)
        except Exception as e:
            print("Folder does not exists.")

        os.mkdir(base_dir) # create a folder so a zip can be stored inside it
        self.model.save(os.path.join(base_dir, self._model_name))

    def load_model(model_name):
        return load_model(os.path.join(config.models_path, model_name))

    def evaluate_existing_model(model_name, dataset, num_of_outputs):
        model = ANN.load_model(model_name)

        X, y = dataset.values[:, :-num_of_outputs], dataset.values[:, -num_of_outputs]
        _, X_test, _, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

        mse, mae = model.evaluate(X_test, y_test, verbose = 0)
        return mse, mae

    def _init_default_hyperparameters(self):
        if self._optimizer is None:
            self._optimizer = 'Adam'
        if self._batch_size is None:
            self._batch_size = 32
        if self._num_of_epochs is None:
            self._num_of_epochs = 5
        if self._activation is None:
            self._activation = 'sigmoid'
        if self._output_activation is None:
            self._output_activation = 'sigmoid'