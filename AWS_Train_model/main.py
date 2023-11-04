import shutil
from warnings import catch_warnings
import config
import os
from database import Db
from models.ANNModel import ANN
from fastapi import FastAPI

app = FastAPI()

@app.post("/trainMlModel")
async def train_ml_model(model_name: str):
    
    try:
        optimizer = 'adam'
        batch_size = 32
        num_of_epochs = 100
        activation = 'relu'
        output_activation = 'linear'

        dataset_full_name = model_name +".csv"
        annModel = ANN(optimizer, batch_size, num_of_epochs, activation, output_activation, dataset_full_name)

        dataset_download_path = os.path.join(config.datasets_path, dataset_full_name)
        Db.download_from_s3(dataset_download_path, dataset_full_name)

        annModel.train_model(dataset_download_path, 1)
    except Exception as e:
        print("Request already fired by S3.")

@app.post("/predictValues")
async def predict_values(model_name: str):
    
    try:
        model_zip_name = model_name +".zip"
        dataset_name = model_name +".csv"

        model_download_path = os.path.join(config.models_path, model_zip_name)
        dataset_download_path = os.path.join(config.datasets_path, dataset_name)

        Db.download_from_s3(local_destination = model_download_path, aws_path = model_zip_name) # download model
        Db.download_from_s3(local_destination = dataset_download_path, aws_path = dataset_name) # download csv

        destination_dir = os.path.join(config.models_path, model_name)
        shutil.unpack_archive(model_download_path, destination_dir)

        model = ANN.load_model(os.path.join(model_name, model_name))

        values = ANN.predict(model = model, dataset = dataset_download_path, num_of_outputs = 1)
        return {"predictions": values[:, 0].tolist()}

    except Exception as e:
        print("Model not found.")

@app.post("/delete_model")
async def delete_model(model_name: str):
    
    try:
        model_zip = model_name +".zip"
        primary_key = model_name +".csv"

        Db.delete_from_dynamoDB(primary_key)
        Db.delete_from_s3(model_zip)

    except Exception as e:
        print("Model not found.")
    