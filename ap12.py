
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# LSTM model training
def train_lstm_model(df, look_back=12, forecast_horizon=6):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.resample('MS').sum()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - look_back - forecast_horizon + 1):
        X.append(scaled[i:i+look_back])
        y.append(scaled[i+look_back:i+look_back+forecast_horizon])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1]))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(look_back, 1)),
        tf.keras.layers.Dense(forecast_horizon)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)

    return model, scaler, df

# Forecast using trained model
def forecast_lstm(model, scaler, df, look_back=12, forecast_horizon=6):
    values = scaler.transform(df)
    last_seq = values[-look_back:]
    input_seq = last_seq.reshape((1, look_back, 1))
    forecast_scaled = model.predict(input_seq)
    forecast = scaler.inverse_transform(forecast_scaled).flatten()

    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='MS')

    forecast_df = pd.DataFrame({'Forecasted Influx': forecast}, index=forecast_dates)
    return forecast_df

# Updated YoY adjustment logic
def adjust_by_yoy_trend_v2(df_input, forecast_df):
    df_input = df_input.copy()
    df_input['Date'] = pd.to_datetime(df_input['Date'])
    df_input['Year'] = df_input['Date'].dt.year
    df_input['Month'] = df_input['Date'].dt.month

    forecast_df = forecast_df.copy()
    forecast_df['Date'] = forecast_df.index
    forecast_df['Month'] = forecast_df['Date'].dt.month
    forecast_df['Year'] = forecast_df['Date'].dt.year

    adjusted_forecasts = []

    for _, row in forecast_df.iterrows():
        month = row['Month']
        year = row['Year']
        base_forecast = row['Forecasted Influx']

        # Gather influx values for the same month from past years
        past_years = df_input[df_input['Month'] == month].groupby('Year')['Influx'].mean().sort_index()

        if len(past_years) >= 2:
            pct_changes = past_years.pct_change().dropna()
            avg_change = pct_changes[-2:].mean()
            adjusted_value = base_forecast * (1 + avg_change)
        else:
            adjusted_value = base_forecast

        adjusted_forecasts.append(adjusted_value)

    forecast_df['Adjusted Forecast'] = adjusted_forecasts
    forecast_df.set_index('Date', inplace=True)
    return forecast_df[['Forecasted Influx', 'Adjusted Forecast']]

# Streamlit app
st.title("📈 Monthly Influx Forecast with LSTM + YoY Adjustment")

uploaded_file = st.file_uploader("Upload an Excel file with Date and Influx columns", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)

    if 'Date' not in df_input.columns or 'Influx' not in df_input.columns:
        st.error("Excel file must contain 'Date' and 'Influx' columns.")
    else:
        st.write("Raw Input Data", df_input.head())

        model, scaler, df_lstm_input = train_lstm_model(df_input)
        forecast_df_monthly = forecast_lstm(model, scaler, df_lstm_input)
        forecast_df_monthly = adjust_by_yoy_trend_v2(df_lstm_input.reset_index(), forecast_df_monthly)

        st.subheader("📊 Forecasted Monthly Influx")
        st.dataframe(forecast_df_monthly)

        st.line_chart(forecast_df_monthly)

        # Download
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=True, sheet_name='Forecast')
            return output.getvalue()

        st.download_button(
            label="📥 Download Forecast as Excel",
            data=to_excel(forecast_df_monthly),
            file_name='monthly_forecast.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
