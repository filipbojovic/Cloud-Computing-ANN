from fileinput import filename
import os
import mysql.connector
import config
from models.fastAPIModel import ModelAPI
import boto3

s3 = boto3.client("s3")
dynamo = boto3.client("dynamodb")

class Db:

    @staticmethod
    def upload_to_s3(local_file_source, aws_path):
        s3.upload_file(Filename = local_file_source, Bucket = config.bucket_name, Key = aws_path) # "Key" - how it will be saved on AWS. filename is local path

    @staticmethod
    def download_from_s3(local_destination, aws_path):
        s3.download_file(Bucket = config.bucket_name, Key = aws_path, Filename = local_destination)
    
    @staticmethod
    def add_model_to_dynamoDB(dataset_name, mse, mae):
        dynamo.put_item(
            TableName = config.dynamodb_table_name,
            Item = {
                'fileName': {'S': dataset_name},
                'mse': {'S': str(mse)},
                'mae': {'S': str(mae)}
            }
        )
    
    @staticmethod
    def delete_from_dynamoDB(primary_key):
        table = boto3.resource("dynamodb").Table(config.dynamodb_table_name)
        table.delete_item(
            Key = {'fileName': primary_key})
    
    @staticmethod
    def delete_from_s3(file_to_delete):
        s3.delete_object(Bucket = config.bucket_name, Key = file_to_delete)