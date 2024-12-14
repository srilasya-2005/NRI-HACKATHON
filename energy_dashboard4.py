import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Title of the Streamlit app
st.markdown("""
    <style>
        .title {
            color: white;
            background-color: #a8d0e6;  /* Light Blue */
            padding: 10px;
            border-radius: 5px;
        }
        .category-select {
            background-color: #f4f4f9;
            padding: 10px;
            border-radius: 5px;
        }
        .data-preview {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
        }
        .chart {
            background-color: #f4f4f9;
            border-radius: 5px;
            padding: 10px;
        }
        .alert-recommendation {
            background-color: #ffefd5;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<h1 class="title">Energy Consumption Forecast Dashboard</h1>', unsafe_allow_html=True)

# Add a search bar with categories
category = st.selectbox(
    "Select a Category",
    ["Household", "Industries", "Weather", "Appliance Use", "Organization"],
    key="category_select"
)

# Display the selected category
st.write(f"### Selected Category: {category}")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Parse the time column and set it as the index
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    data.set_index('time', inplace=True)

    # Ensure the index is sorted
    data.sort_index(inplace=True)

    # Display data preview
    st.markdown('<div class="data-preview">', unsafe_allow_html=True)
    st.write("### Data Preview")
    st.write(data.head())
    st.markdown('</div>', unsafe_allow_html=True)

    # Extract the first 4 months of data
    data_4_months = data.loc['2011-11-11':'2012-03-10']

    # Drop rows with missing values
    data_4_months = data_4_months.dropna()

    # Encode categorical features
    label_encoder = LabelEncoder()
    data_4_months['summary'] = label_encoder.fit_transform(data_4_months['summary'])
    data_4_months['icon'] = label_encoder.fit_transform(data_4_months['icon'])

    # Define the features and target
    features = data_4_months.drop(columns=['temperature'])
    target = data_4_months['temperature']

    # Ensure all features are numeric
    features = pd.get_dummies(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    feature_names = features.columns

    # Plot feature importances
    fig, ax = plt.subplots()
    ax.barh(feature_names, importances, color='green')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Using Random Forest')
    st.pyplot(fig)

    # Create a feature importance bar chart
    fig_importance = px.bar(
        x=importances,
        y=feature_names,
        orientation='h',
        labels={'x': 'Feature Importance', 'y': 'Features'},
        title='Feature Importance Using Random Forest'
    )
    st.plotly_chart(fig_importance)

    # Ensure only numeric columns are used for aggregation
    numeric_data = data.select_dtypes(include=[np.number])

    # Aggregate data for different granularities
    daily_data = numeric_data.resample('D').mean()
    hourly_data = numeric_data.resample('H').mean()
    monthly_data = numeric_data.resample('M').mean()

    # Display aggregated data as graphs
    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.write("### Daily Data")
    fig_daily = px.line(daily_data, labels={'value': 'Value', 'index': 'Date'}, title='Daily Data')
    st.plotly_chart(fig_daily)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.write("### Hourly Data")
    fig_hourly = px.line(hourly_data, labels={'value': 'Value', 'index': 'Date'}, title='Hourly Data')
    st.plotly_chart(fig_hourly)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.write("### Monthly Data")
    fig_monthly = px.line(monthly_data, labels={'value': 'Value', 'index': 'Date'}, title='Monthly Data')
    st.plotly_chart(fig_monthly)
    st.markdown('</div>', unsafe_allow_html=True)

    # Function to generate alerts based on consumption
    def generate_alerts(data, threshold):
        high_consumption = data[data['temperature'] > threshold]
        if not high_consumption.empty:
            return f"Alert: High energy consumption detected on {high_consumption.index.date[0]}"
        return "No alerts"

    # Function to provide recommendations
    def get_recommendations(data):
        avg_temp = data['temperature'].mean()
        if avg_temp > 25:
            return "Recommendation: Consider using cooling systems more efficiently."
        elif avg_temp < 15:
            return "Recommendation: Consider using heating systems more efficiently."
        return "Recommendation: Energy usage is optimal."

    # Generate alerts and recommendations
    alert_message = generate_alerts(daily_data, threshold=30)
    recommendation_message = get_recommendations(daily_data)

    st.markdown('<div class="alert-recommendation">', unsafe_allow_html=True)
    st.write("### Alerts")
    st.write(alert_message)
    st.write("Example Alerts:")
    st.write("1. Alert: High energy consumption detected on 2011-12-15.")
    st.write("2. Alert: High energy consumption detected on 2012-01-20.")
    st.write("3. Alert: High energy consumption detected on 2012-02-10.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="alert-recommendation">', unsafe_allow_html=True)
    st.write("### Recommendations")
    st.write(recommendation_message)
    st.write("Example Recommendations:")
    st.write("1. Recommendation: Consider using cooling systems more efficiently.")
    st.write("2. Recommendation: Consider using heating systems more efficiently.")
    st.write("3. Recommendation: Optimize the use of electrical appliances during peak hours.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Forecasting with ARIMA
    def forecast_arima(data, steps=30):
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    # Forecasting with LSTM
    def forecast_lstm(data, steps=30):
        data_values = data.values
        data_values = data_values.reshape((-1, 1))

        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_values = scaler.fit_transform(data_values)

        # Prepare data for LSTM
        generator = TimeseriesGenerator(data_values, data_values, length=10, batch_size=1)

        # Define LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Train the model
        lstm_model.fit(generator, epochs=10)

        # Generate predictions
        predictions = []
        current_batch = data_values[-10:].reshape((1, 10, 1))

        for i in range(steps):
            current_pred = lstm_model.predict(current_batch)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)
        return predictions.flatten()

    # Forecast using ARIMA
    arima_forecast = forecast_arima(daily_data['temperature'], steps=30)
    st.write("### ARIMA Forecast")
    arima_dates = pd.date_range(start=daily_data.index[-1], periods=30, freq='D')
    fig_arima = px.line(x=arima_dates, y=arima_forecast, labels={'x': 'Date', 'y': 'ARIMA Forecast'})
    st.plotly_chart(fig_arima)

    # Forecast using LSTM
    lstm_forecast = forecast_lstm(daily_data['temperature'], steps=30)
    st.write("### LSTM Forecast")
    lstm_dates = pd.date_range(start=daily_data.index[-1], periods=30, freq='D')
    fig_lstm = px.line(x=lstm_dates, y=lstm_forecast, labels={'x': 'Date', 'y': 'LSTM Forecast'})
    st.plotly_chart(fig_lstm)
else:
    st.write("Please upload a CSV file.")
