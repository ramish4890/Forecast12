import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os, random
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split





# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="Influx Forecasting Dashboard")
st.title("Influx Forecasting Dashboard")
st.write("Upload a single Excel file containing all required data sheets to generate various influx forecasts.")

# --- Reproducibility (for TensorFlow) ---
@st.cache_resource # Cache the model and environment settings
def set_seeds():
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warnings
    tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logger warnings
set_seeds()


# --- Function for Revised Forecast (Your original code) ---
def run_revised_forecast(uploaded_file):
    st.header("Revised Forecast (Hourly Proportions based)")
    st.info("Starting Revised Forecast generation...")
    try:
        # Expected sheets: 'hourly', 'mmf'
        df_hourly = pd.read_excel(uploaded_file, sheet_name="hourly")
        df_forecast = pd.read_excel(uploaded_file, sheet_name='mmf')

        # Step 4: Preprocess hourly data
        df_hourly['Date'] = pd.to_datetime(df_hourly['Date'])
        df_hourly['Weekday'] = df_hourly['Date'].dt.day_name()

        # Step 5: Group by Pod, Weekday, Hour and compute proportions
        grouped = df_hourly.groupby(['Pod', 'Weekday', 'Hour'])['Influx'].sum().reset_index()
        totals = grouped.groupby(['Pod', 'Weekday'])['Influx'].sum().reset_index()
        totals.rename(columns={'Influx': 'TotalInflux'}, inplace=True)
        merged = pd.merge(grouped, totals, on=['Pod', 'Weekday'])
        merged['HourlyProportion'] = merged['Influx'] / merged['TotalInflux']

        # Step 6: Outlier filtering but keep all weekdays
        cleaned_data = []
        for pod in merged['Pod'].unique():
            pod_data = merged[merged['Pod'] == pod]
            for weekday in pod_data['Weekday'].unique():
                sub_df = pod_data[pod_data['Weekday'] == weekday].copy()
                q1 = sub_df['HourlyProportion'].quantile(0.25)
                q3 = sub_df['HourlyProportion'].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                sub_df = sub_df[(sub_df['HourlyProportion'] >= lower) & (sub_df['HourlyProportion'] <= upper)]
                cleaned_data.append(sub_df)
        final_df = pd.concat(cleaned_data, ignore_index=True)

        # Step 7: Pivot table (optional inspection)
        pivot = final_df.pivot_table(index=['Pod', 'Weekday'],
                                     columns='Hour',
                                     values='HourlyProportion',
                                     fill_value=0)

        # Step 8: Load and clean forecast (from 'mmf' sheet)
        df_forecast.columns = ['Month', 'Forecasted Influx']
        df_forecast['Month'] = pd.to_datetime(df_forecast['Month'], errors='coerce')
        df_forecast.dropna(subset=['Month'], inplace=True)
        df_forecast.set_index('Month', inplace=True)
        df_forecast = df_forecast.sort_index()
        df_forecast = df_forecast.asfreq('MS')

        # Step 9: Get pod proportions and weekday distributions (from 'hourly' sheet data)
        daily_df_hourly = df_hourly.groupby(['Date', 'Pod'])['Influx'].sum().reset_index()
        daily_df_hourly['Weekday'] = daily_df_hourly['Date'].dt.weekday

        pod_total = daily_df_hourly.groupby('Pod')['Influx'].sum()
        pod_prop = pod_total / pod_total.sum()

        weekday_total = daily_df_hourly.groupby(['Pod', 'Weekday'])['Influx'].sum()
        weekday_prop = weekday_total.groupby(level=0).apply(lambda x: x / x.sum())

        # Step 10: Monthly pod forecast
        monthly_pod_forecast = pd.DataFrame(index=df_forecast.index)
        for pod in pod_prop.index:
            monthly_pod_forecast[pod] = df_forecast['Forecasted Influx'] * pod_prop[pod]

        # Step 11: Daily pod forecast (corrected logic)
        daily_forecast_all = []
        weekday_name_map = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }

        for month_start in df_forecast.index:
            month_end = month_start + pd.offsets.MonthEnd(0)
            days = pd.date_range(start=month_start, end=month_end, freq='D')
            weekday_indices = [d.weekday() for d in days]
            weekday_counts = pd.Series(weekday_indices).value_counts().to_dict()

            month_data = {'Date': days}
            for pod in pod_prop.index:
                pod_month_total = monthly_pod_forecast.loc[month_start, pod]
                pod_week_props = weekday_prop.loc[pod]
                pod_daily = []

                sum_of_relevant_weekday_props = sum(pod_week_props.get(idx, 0) for idx in weekday_indices)

                for day_idx, day in enumerate(days):
                    weekday_idx = day.weekday()
                    weekday_prop_val = pod_week_props.get(weekday_idx, 0)

                    if sum_of_relevant_weekday_props > 0:
                        daily_val = (pod_month_total * weekday_prop_val) / sum_of_relevant_weekday_props * weekday_counts.get(weekday_idx, 1)
                    else:
                        daily_val = 0
                    pod_daily.append(daily_val)
                month_data[pod] = pod_daily

            daily_forecast_all.append(pd.DataFrame(month_data))

        daily_forecast_df = pd.concat(daily_forecast_all, ignore_index=True)
        daily_forecast_df.set_index('Date', inplace=True)

        # Step 12: Hourly forecast
        hourly_forecast_all = []
        for date, row in daily_forecast_df.iterrows():
            weekday_name = weekday_name_map[date.weekday()]
            for pod in pod_prop.index:
                if pod not in row:
                    continue
                daily_val = row[pod]
                pod_hourly_props = final_df[(final_df['Pod'] == pod) & (final_df['Weekday'] == weekday_name)]

                if not pod_hourly_props.empty:
                    sum_hr_prop = pod_hourly_props['HourlyProportion'].sum()
                    if sum_hr_prop > 0:
                        normalized_hourly_props = pod_hourly_props['HourlyProportion'] / sum_hr_prop
                    else:
                        normalized_hourly_props = pd.Series([0] * len(pod_hourly_props), index=pod_hourly_props.index)

                    for idx, hr_row in pod_hourly_props.iterrows():
                        hour = hr_row['Hour']
                        hr_prop = normalized_hourly_props.loc[idx]
                        hourly_forecast_all.append({
                            'Date': date,
                            'Hour': int(hour),
                            'Pod': pod,
                            'Forecasted Influx': (daily_val * hr_prop)
                        })

        hourly_forecast_df = pd.DataFrame(hourly_forecast_all)
        hourly_forecast_df = hourly_forecast_df.sort_values(by=['Date', 'Hour', 'Pod'])

        st.success("Revised Forecast generated!")

        return df_forecast, monthly_pod_forecast, daily_forecast_df, hourly_forecast_df, pivot

    except Exception as e:
        st.error(f"Error in Revised Forecast: Please ensure the uploaded file contains 'hourly' and 'mmf' sheets with correct data structure. Error: {e}")
        st.exception(e)
        return None, None, None, None, None

# --- Function for Monthly Forecast (New LSTM-based code) ---
def run_monthly_forecast(uploaded_file):
    st.header("Monthly Forecast (LSTM-based)")
    st.info("Starting Monthly Forecast generation...")

    try:

        
        # Example data (replace with your real data)
        # df = pd.read_csv('your_data.csv')
        # Assuming you already have your X, y ready
        
        # Normalize if needed
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Model Architecture
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        model = Sequential([
            LSTM(32, activation='relu', return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(output_dim)
        ])
        
        # Compile
        model.compile(optimizer='adam', loss='mse')
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=4,
            callbacks=[early_stop, lr_schedule],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Evaluate
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
# === STEP 7: Adjust forecast using historical trends (keep same column name) ===
                
        adjusted_forecast = []
        avg_changes = []
        
        # Get latest year in historical input — treat it as "current year"
        latest_year = df_lstm_input.index.max().year
        
        for forecast_month in forecast_df_monthly.index:
            month = forecast_month.month
        
            if month not in [4,5,6,7,8,9,10, 11, 12]:  # Oct–Dec only
                adjusted = forecast_df_monthly.loc[forecast_month, 'Forecasted Influx']
                avg_change = 0
            else:
                historical_years = [1, 2, 3]
                past_values = []
        
                for i in historical_years:
                    past_year = latest_year - i
                    past_date = pd.Timestamp(year=past_year, month=month, day=1)
        
                    if past_date in df_lstm_input.index:
                        past_values.append(df_lstm_input.loc[past_date, 'Influx'])
        
                # Calculate avg % change if we have 2+ values
                if len(past_values) >= 2:
                    diffs = []
                    for prev, curr in zip(past_values[1:], past_values[:-1]):
                        if prev != 0:
                            diffs.append((curr - prev) / prev)
        
                    avg_change = np.mean(diffs) if diffs else 0
                    print(f"Avg % change for {forecast_month.strftime('%Y-%m')}: {avg_change:.2%}")
        
                    # Compare current forecast vs last year’s influx
                    ref_date = pd.Timestamp(year=latest_year - 1, month=month, day=1)
                    if ref_date in df_lstm_input.index:
                        last_year_influx = df_lstm_input.loc[ref_date, 'Influx']
                        current_forecast = forecast_df_monthly.loc[forecast_month, 'Forecasted Influx']
        
                        # Actual forecasted % change
                        if last_year_influx != 0:
                            actual_change = (current_forecast - last_year_influx) / last_year_influx
                        else:
                            actual_change = 0
        
                        # Difference between desired and actual change
                        gap = avg_change - actual_change
        
                        # Adjust forecast further to meet full historical trend
                        adjusted = current_forecast * (1 + gap)
                    else:
                        adjusted = forecast_df_monthly.loc[forecast_month, 'Forecasted Influx']
                else:
                    avg_change = 0
                    adjusted = forecast_df_monthly.loc[forecast_month, 'Forecasted Influx']
        
            adjusted_forecast.append(adjusted)
            avg_changes.append(avg_change)
        
        # Store back
        forecast_df_monthly['Adjusted Forecast'] = adjusted_forecast
        forecast_df_monthly['YOY % Change'] = [f"{c*100:.2f}%" for c in avg_changes]


































        
        st.info("Monthly influx forecast generated.")

        # === STEP 7: Daily Pod Forecast ===
        # Use the 'daily' sheet from the same uploaded file
        daily_df_monthly_input = pd.read_excel(uploaded_file, sheet_name="daily")
        daily_df_monthly_input['Date'] = pd.to_datetime(daily_df_monthly_input['Date'])
        daily_df_monthly_input['Weekday'] = daily_df_monthly_input['Date'].dt.weekday
        min_days = daily_df_monthly_input['Weekday'].value_counts().min()
        balanced = daily_df_monthly_input.groupby('Weekday').apply(lambda x: x.sample(min_days, random_state=42)).reset_index(drop=True)

        pod_total = balanced.groupby('Pod')['Influx'].sum()
        pod_prop = pod_total / pod_total.sum()

        monthly_pod_forecast = pd.DataFrame(index=future_dates)
        for pod in pod_prop.index:
            monthly_pod_forecast[pod] = forecast_df_monthly['Forecasted Influx'] * pod_prop[pod]

        weekday_total = balanced.groupby(['Pod', 'Weekday'])['Influx'].sum()
        weekday_prop = weekday_total.groupby(level=0).apply(lambda x: x / x.sum())

        daily_forecast_all = []
        for month_start in future_dates:
            month_end = month_start + pd.offsets.MonthEnd(0)
            days = pd.date_range(start=month_start, end=month_end, freq='D')
            weekdays_in_month = [d.weekday() for d in days]
            month_data = {'Date': days}
            for pod in pod_prop.index:
                pod_month_total = monthly_pod_forecast.loc[month_start, pod]
                pod_week_props = weekday_prop.loc[pod]
                pod_daily = []

                sum_of_relevant_weekday_props = sum(pod_week_props.get(idx, 0) for idx in weekdays_in_month)
                weekday_counts_in_month = pd.Series(weekdays_in_month).value_counts()

                for day in days:
                    weekday_idx = day.weekday()
                    weekday_prop_val = pod_week_props.get(weekday_idx, 0)

                    if sum_of_relevant_weekday_props > 0:
                        daily_val = (pod_month_total * weekday_prop_val / sum_of_relevant_weekday_props) * weekday_counts_in_month.get(weekday_idx, 0)
                    else:
                        daily_val = 0
                    pod_daily.append(daily_val)

                month_data[pod] = pod_daily

            daily_forecast_all.append(pd.DataFrame(month_data))

        daily_forecast_df = pd.concat(daily_forecast_all, ignore_index=True)
        daily_forecast_df.set_index('Date', inplace=True)
        st.info("Daily pod forecast generated.")

        # === STEP 8: Hourly Forecast ===
        # Use the 'hourly' sheet from the same uploaded file
        hourly_df_monthly_input = pd.read_excel(uploaded_file, sheet_name="hourly")
        hourly_df_monthly_input['Date'] = pd.to_datetime(hourly_df_monthly_input['Date'])
        hourly_df_monthly_input['Weekday'] = hourly_df_monthly_input['Date'].dt.day_name()
        grouped = hourly_df_monthly_input.groupby(['Pod', 'Weekday', 'Hour'])['Influx'].sum().reset_index()
        totals = grouped.groupby(['Pod', 'Weekday'])['Influx'].sum().reset_index().rename(columns={'Influx': 'TotalInflux'})
        merged = pd.merge(grouped, totals, on=['Pod', 'Weekday'])
        merged['HourlyProportion'] = merged['Influx'] / merged['TotalInflux']

        hourly_forecast_all = []
        weekday_name_map = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        for date, row in daily_forecast_df.iterrows():
            weekday_name = weekday_name_map[date.weekday()]
            for pod in pod_prop.index:
                if pod not in row:
                    continue
                daily_val = row[pod]
                pod_hourly_props = merged[(merged['Pod'] == pod) & (merged['Weekday'] == weekday_name)]

                if not pod_hourly_props.empty:
                    sum_hr_prop = pod_hourly_props['HourlyProportion'].sum()
                    if sum_hr_prop > 0:
                        normalized_hourly_props = pod_hourly_props['HourlyProportion'] / sum_hr_prop
                    else:
                        normalized_hourly_props = pd.Series([0] * len(pod_hourly_props), index=pod_hourly_props.index)

                    for idx, hr_row in pod_hourly_props.iterrows():
                        hour = hr_row['Hour']
                        hr_prop = normalized_hourly_props.loc[idx]
                        hourly_forecast_all.append({
                            'Date': date,
                            'Hour': int(hour),
                            'Pod': pod,
                            'Forecasted Influx': (daily_val * hr_prop)
                        })

        hourly_forecast_df = pd.DataFrame(hourly_forecast_all).sort_values(by=['Date', 'Hour', 'Pod'])
        st.success("Hourly pod forecast generated!")

        return forecast_df_monthly, monthly_pod_forecast, daily_forecast_df, hourly_forecast_df, None # No pivot for monthly forecast

    except Exception as e:
        st.error(f"Error in Monthly Forecast: Please ensure the uploaded file contains 'monthly forecast', 'daily', and 'hourly' sheets with correct data structure. Error: {e}")
        st.exception(e)
        return None, None, None, None, None


# --- Streamlit UI Layout ---

# File Uploader for the input data
st.sidebar.header("Upload Data")
st.sidebar.markdown("**Please upload a single Excel file containing ALL the necessary sheets:**")
st.sidebar.markdown("- For **Revised Forecast**: sheets hourly, mmf")
st.sidebar.markdown("- For **Monthly Forecast**: sheets monthly forecast, daily, hourly")
st.sidebar.markdown("*(Note: The hourly sheet will be used by both forecasting methods.)*")

uploaded_file = st.sidebar.file_uploader("Choose your Excel file", type="xlsx", key="main_upload")

forecast_type = st.sidebar.radio(
    "Select Forecast Type:",
    ("Revised Forecast", "Monthly Forecast (LSTM)")
)

output_filename_default = "forecast_output.xlsx"
if forecast_type == "Revised Forecast":
    output_filename_default = "Revised_Forecast_Output.xlsx"
elif forecast_type == "Monthly Forecast (LSTM)":
    output_filename_default = "Monthly_Forecast_Output.xlsx"

output_filename = st.sidebar.text_input("Output Excel filename:", output_filename_default)

if uploaded_file is not None:
    results_df_forecast = None
    results_monthly_pod = None
    results_daily_pod = None
    results_hourly_pod = None
    results_pivot = None

    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Generating forecast... This may take a moment, especially for LSTM training."):
            if forecast_type == "Revised Forecast":
                st.write("---")
                results_df_forecast, results_monthly_pod, results_daily_pod, results_hourly_pod, results_pivot = run_revised_forecast(uploaded_file)
            elif forecast_type == "Monthly Forecast (LSTM)":
                st.write("---")
                results_df_forecast, results_monthly_pod, results_daily_pod, results_hourly_pod, results_pivot = run_monthly_forecast(uploaded_file)

        if results_df_forecast is not None:
            st.success("Forecast calculation complete!")
            st.write("---")
            st.subheader("Download Forecast Results")

            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                results_df_forecast.to_excel(writer, sheet_name='Monthly Forecast')
                results_monthly_pod.to_excel(writer, sheet_name='Monthly Pod Forecast')
                results_daily_pod.to_excel(writer, sheet_name='Daily Pod Forecast')
                results_hourly_pod.to_excel(writer, sheet_name='Hourly Pod Forecast', index=False)
                if results_pivot is not None: # Only save pivot for Revised Forecast
                    results_pivot.to_excel(writer, sheet_name='Hourly Proportions (7 Days)')

            excel_buffer.seek(0)

            st.download_button(
                label="Download Forecast Excel",
                data=excel_buffer,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success(f"File '{output_filename}' ready for download.")
            st.write("---")

            # Display previews (optional)
            st.subheader("Preview of Forecasted Data")
            st.markdown("**(Please scroll to view all tables)**")
            st.markdown("### Monthly Forecast:")
            st.dataframe(results_df_forecast)
            st.markdown("### Monthly Pod Forecast:")
            st.dataframe(results_monthly_pod)
            st.markdown("### Daily Pod Forecast:")
            st.dataframe(results_daily_pod)
            st.markdown("### Hourly Pod Forecast:")
            st.dataframe(results_hourly_pod)
            if results_pivot is not None:
                st.markdown("### Hourly Proportions (7 Days - Used in Revised Forecast):")
                st.dataframe(results_pivot)

else:
    st.info("Please upload an Excel file to start forecasting.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Your Name/Company")
