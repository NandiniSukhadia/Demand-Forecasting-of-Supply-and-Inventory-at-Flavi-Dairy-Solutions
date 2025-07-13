def create_forecasting_section(df):
    """Create the forecasting section"""
    st.markdown("## 🔮 Demand Forecasting")

    # Target selection
    target_options = {
        'Total Demand': 'Total_Demand',
        'Milk Supply': 'Milk_Supply_Liters',
        'Milk 500ml Demand': 'Milk_500ml_Demand',
        'Milk 1L Demand': 'Milk_1L_Demand',
        'Butter Demand': 'Butter_Demand',
        'Cheese Demand': 'Cheese_Demand',
        'Yogurt Demand': 'Yogurt_Demand'
    }

    selected_target = st.selectbox("Select target variable for forecasting:", list(target_options.keys()))
    target_col = target_options[selected_target]

    # Prepare features
    X, y, df_processed = prepare_features(df.copy(), target_col)

    # Button to trigger training
    if st.button("🚀 Train Forecasting Models"):
        with st.spinner("Training models..."):
            results, X_train, X_test, y_train, y_test = train_models(X, y)

        # Show model performance
        st.markdown("### 📊 Model Performance Comparison")

        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'R²': result['r2']
            })

        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)

        # Best model selection
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]['model']
        best_scaler = results[best_model_name]['scaler']

        st.success(f"🎯 Best Model: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.2f})")

        # --- 🔮 Future Forecasting Block ---
        st.markdown("### 🔮 Future Forecast")

        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Quarter'] = future_df['Date'].dt.quarter
        future_df['Year'] = future_df['Date'].dt.year
        future_df['sin_day'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['cos_day'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['sin_month'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['cos_month'] = np.cos(2 * np.pi * future_df['Month'] / 12)

        # Lag features from last known values
        last_values = df[target_col].tail(30).values
        for i in range(len(future_df)):
            if i < len(last_values):
                future_df.loc[i, f'{target_col}_lag_1'] = last_values[-(i + 1)]
                if i < len(last_values) - 7:
                    future_df.loc[i, f'{target_col}_lag_7'] = last_values[-(i + 8)]
                if i < len(last_values) - 14:
                    future_df.loc[i, f'{target_col}_lag_14'] = last_values[-(i + 15)]
                if i < len(last_values) - 30:
                    future_df.loc[i, f'{target_col}_lag_30'] = last_values[-(i + 31)]

        future_df = future_df.fillna(df[target_col].mean())

        # Ensure future features match training set
        future_features = [col for col in future_df.columns if col != 'Date']
        for col in X.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X.columns]

        # Predict
        if best_scaler:
            X_future_scaled = best_scaler.transform(future_df)
            future_predictions = best_model.predict(X_future_scaled)
        else:
            future_predictions = best_model.predict(future_df)

        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df[target_col], name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'{selected_target} Forecast (Next 30 Days)', xaxis_title='Date', yaxis_title=selected_target, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions})
        forecast_df['Forecast'] = forecast_df['Forecast'].round(2)
        st.markdown("### 📋 Forecast Details")
        st.dataframe(forecast_df, use_container_width=True)
