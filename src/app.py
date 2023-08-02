import os
import joblib
import pandas as pd
import streamlit as st
import boto3
import botocore
from io import BytesIO
from dotenv import load_dotenv

from src.utils import read_yaml_file

st.title("Customer Engagement Prediction")

st.write("Instructions")
st.write("This is a simple front end to enable users to make predictions on possible.")
st.write("Simply fill in the details of the profile of cluster and a prediction will be created.")

### Frontend Data Collection ###
col1, col2 = st.columns(2)

households = col1.number_input("Number of households in cluster", value=344, step=1)
total_rooms = col1.number_input("Number of rooms in cluster", value=1893, step=1)
housing_median_age = col1.number_input("Median age of house in cluster?", value=16, step=1)
median_income = col2.number_input("Median income (USD) in cluster", value=5.225)
ocean_proximity = col2.radio(
    "Please select ocean proximity?", ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND')
)

## Backend functions
def make_df(
    households: int,
    total_rooms: int,
    housing_median_age: float,
    median_income: float,
    ocean_proximity: str,
) -> pd.DataFrame:
    """Function creates a prediction dataframe to be predicted by the model
    Args:
        households (int): Number of households in cluster
        total_rooms (int): Total Number of rooms in cluster
        housing_median_age (int): Median age of house in cluster
        median_income (int): Median income in cluster
        ocean_proximity (str): Ocean proximity type
    Returns:
        pd.DataFrame: dataframe for prediction
    """
    pred_row = pd.DataFrame(
        {
            "households": int(households),
            "total_rooms": int(total_rooms),
            "housing_median_age": int(housing_median_age),
            "median_income": float(median_income),
            "ocean_proximity": str(ocean_proximity),
        },
        index=[0],
    )

    return pred_row


@st.cache_data
def load_model():
    """Loads the model and returns it for inference
    Returns:
        sklearn.model: model as per config file setting
    """
    path_to_dotenv_file = os.path.abspath(os.path.join(os.getcwd(), "docker/.env"))
    print(path_to_dotenv_file)
    if os.path.isfile(path_to_dotenv_file):
        load_dotenv(dotenv_path=path_to_dotenv_file)
        MINIO_LOCATION = os.environ.get("MLFLOW_S3_ENDPOINT_URL") 
        ACCESSKEY = os.environ.get("MINIO_ACCESS_KEY" ) 
        SECRETKEY = os.environ.get("MINIO_SECRET_KEY" ) 

        config = read_yaml_file()
        model_path = config["streamlit"]["model_path"]
        model_name ="/".join(model_path.split("/")[3:])

        s3 = boto3.client('s3', 
                aws_access_key_id = ACCESSKEY,
                aws_secret_access_key = SECRETKEY,
                aws_session_token=None,
                endpoint_url=MINIO_LOCATION,
                config=boto3.session.Config(signature_version='s3v4'),
                verify = False)

        if s3.list_buckets()["Buckets"][0]["Name"] == "mlflow":
            response = s3.get_object(Bucket="mlflow", Key=model_name)
            with BytesIO(response['Body'].read()) as f:
                f.seek(0)
                model = joblib.load(f)
            return model
        else:
            print("Bucket Mlflow not found")


model = load_model()

st.subheader("Make a Prediction")
predict = st.button("Click me!")

if predict:
    pred_df = make_df(
        households,
        total_rooms,
        housing_median_age,
        median_income,
        ocean_proximity,
    )
    prediction = model.predict(pred_df)

    st.write(
        f"The predicted Median CES score is {prediction.tolist()[0]:.2f}."
    )