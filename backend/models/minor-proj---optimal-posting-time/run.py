import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the model with custom objects
model = load_model("lstm_model_2.h5", custom_objects={"mse": MeanSquaredError()})

# Function to generate data for the given date range
def generate_data_for_date_range(start_date, end_date , input_values , mean_values):
    # Generate a range of timestamps between the start and end dates
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create constant values for Likes, Comments, Shares, Impressions, Reach based on the provided mean_values
    data = {
        "Likes": mean_values["mean_Likes"] * len(timestamps),  
        "Comments": mean_values["mean_comments"] * len(timestamps),  
        "Shares": mean_values["mean_Shares"] * len(timestamps),  
        "Impressions": mean_values["mean_Impressions"] * len(timestamps),  
        "Reach": mean_values["mean_reach"] * len(timestamps),  
        "Audience Age": input_values['Audience Age']*len(timestamps),
        "Platform_Instagram": input_values['Platform_Instagram']*len(timestamps),
        "Platform_LinkedIn": input_values['Platform_LinkedIn']*len(timestamps),
        "Platform_Twitter": input_values['Platform_Twitter']*len(timestamps),
        "Post Type_Link": input_values['Post Type_Link']*len(timestamps),
        "Post Type_Video": input_values['Post Type_Video']*len(timestamps),
        "Audience Gender_Male": input_values['Audience Gender_Male']*len(timestamps),
        "Audience Gender_Other": input_values['Audience Gender_Other']*len(timestamps),
        "Sentiment_Neutral":input_values['Sentiment_Neutral']*len(timestamps),
        "Sentiment_Positive": input_values['Sentiment_Positive']*len(timestamps),
    }

    # Create a DataFrame with the generated data
    df = pd.DataFrame(data, index=timestamps)

    # Extract features from the timestamp
    df['Day'] = df.index.day
    df['Minute'] = df.index.minute
    df['Second'] = df.index.second
    df['Day of Week'] = df.index.dayofweek
    df['Is Weekend'] = df['Day of Week'].apply(lambda x: 1 if x >= 5 else 0)

    # Create cyclic features for Month and Hour
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Reset index and drop the timestamp column
    df = df.reset_index(drop=True)

    return df

# Example input dates (you can change these as needed)
start_date = "2023-01-01"
end_date = "2023-01-30"
input_values = {
    "Audience Age": 28, 
        "Platform_Instagram": 1,
        "Platform_LinkedIn": 1,
        "Platform_Twitter": 0,
        "Post Type_Link": 0,
        "Post Type_Video": 1,
        "Audience Gender_Male": 0,
        "Audience Gender_Other": 1,
        "Sentiment_Neutral": 1,
        "Sentiment_Positive": 0
}

# mean values , will always remain constant
mean_values = {
    "mean_Likes" :  0.49939624000000005,
    "mean_Shares" : 0.50059255,
    "mean_Impressions" : 0.4986254511111111,
    "mean_comments" : 0.4993984,
    "mean_reach" : 0.5003377733333333 
}


# Generate the data for the given date range with mean values
df = generate_data_for_date_range(start_date, end_date, input_values , mean_values)

# Define the target and features
target = "Engagement Rate"  # Assuming you have this column in your full dataset
features = [col for col in df.columns if col != target]

# Scale the input features (use the same scaler that was used during training)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Prepare input data (use the last 1 timestep, as time_steps is set to 1 here)
time_steps = 1
input_data = []

# Add dummy data to meet the time_steps requirement
for i in range(time_steps, len(scaled_data)):
    input_data.append(scaled_data[i-time_steps:i])

input_data = np.array(input_data)

predicted_engagement_rate = model.predict(input_data)

# Assuming `predicted_engagement_rate` is a numpy array
predicted_engagement_rate = predicted_engagement_rate.flatten()  # Flatten in case it's a multidimensional array

# Recreate the timestamps for the generated data
timestamps = pd.date_range(start=start_date, end=end_date, freq='D')

timestamps = timestamps[1:]

# Create a DataFrame to combine timestamps and predicted engagement rates
results = pd.DataFrame({
    "Timestamp": timestamps,
    "Predicted Engagement Rate": predicted_engagement_rate
})

# Print the results in order
results = results.sort_values(by="Predicted Engagement Rate", ascending=False).reset_index(drop=True)
print(results)



