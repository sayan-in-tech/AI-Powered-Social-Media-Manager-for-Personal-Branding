import pandas as pd
from datetime import datetime
import joblib
import os

# Ensure the model path is correct
model_path = os.path.join(os.path.dirname(__file__), 'twitter_hashtag_predictor_0.1v.pkl')

# Load the model
try:
    model = joblib.load(model_path)
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

def predict_single_hashtag(hashtag, model=model, current_datetime=None):
    if current_datetime is None:
        current_datetime = datetime.now()

    month_name = current_datetime.strftime('%B')
    weekday_name = current_datetime.strftime('%A')

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    weekday_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }

    month = month_map.get(month_name)
    weekday = weekday_map.get(weekday_name)

    # Ensure that data format matches the model's expected input
    new_data = pd.DataFrame({
        'trend_name': [hashtag],
        'hour': [current_datetime.hour],
        'day': [current_datetime.day],
        'month': [month],
        'weekday': [weekday]
    })

    # Predict using the model
    try:
        prediction = model.predict_proba(new_data)[0][1]
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

    return prediction