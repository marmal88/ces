import joblib
import numpy as np
import pandas as pd
import streamlit as st

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
def load_model(model_name: str):
    """Loads the model and returns it for inference
    Args:
        model_name (str): file name of the model
    Returns:
        sklearn.model: model as per config file setting
    """
    loaded_model = joblib.load(f"models/{model_name}")
    return loaded_model


model_name = "lin_reg_pipe.pkl"
model = load_model(model_name)

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