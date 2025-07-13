# Create forecast
st.markdown("### 🔮 Future Forecast")

# Generate future dates
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# Prepare future features
future_df = pd.DataFrame({'Date': future_dates})
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['Month'] = future_df['Date'].dt.month
future_df['Quarter'] = future_df['Date'].dt.quarter
future_df['Year'] = future_df['Date'].dt.year

# Add seasonal features
future_df['sin_day'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
future_df['cos_day'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
future_df['sin_month'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['cos_month'] = np.cos(2 * np.pi * future_df['Month'] / 12)

# Fix: Use last known values for lag features (ensure correct length)
last_known = df[target_col].fillna(method="ffill").tail(30).tolist()
future_df[f'{target_col}_lag_1'] = last_known[-1]
future_df[f'{target_col}_lag_7'] = last_known[-7] if len(last_known) >= 7 else last_known[-1]
future_df[f'{target_col}_lag_14'] = last_known[-14] if len(last_known) >= 14 else last_known[-1]
future_df[f'{target_col}_lag_30'] = last_known[-30] if len(last_known) >= 30 else last_known[-1]

# Fill rolling mean/std using overall training data
for window in [7, 14, 30]:
    mean_val = df[target_col].rolling(window).mean().dropna().mean()
    std_val = df[target_col].rolling(window).std().dropna().mean()
    future_df[f'{target_col}_rolling_mean_{window}'] = mean_val
    future_df[f'{target_col}_rolling_std_{window}'] = std_val

# Match features with training
for col in X.columns:
    if col not in future_df.columns:
        future_df[col] = 0
future_df = future_df[X.columns]

# Scale if needed
if best_scaler:
    future_scaled = best_scaler.transform(future_df)
    forecast = best_model.predict(future_scaled)
else:
    forecast = best_model.predict(future_df)

# Show result
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': np.round(forecast, 2)
})
st.markdown("### 📋 Forecast Details")
st.dataframe(forecast_df, use_container_width=True)
