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

    if st.button("🚀 Train Forecasting Models"):
        with st.spinner("Training models..."):
            results, X_train, X_test, y_train, y_test = train_models(X, y)

        # Display results
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

        # Best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]['model']
        best_scaler = results[best_model_name]['scaler']

        st.success(f"🎯 Best Model: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.2f})")

        # Forecast
        st.markdown("### 🔮 Future Forecast")

        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30, freq='D')
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Quarter'] = future_df['Date'].dt.quarter
        future_df['Year'] = future_df['Date'].dt.year

        # Combine with past data to get lag features
        df_lags = df[['Date', target_col, 'DayOfWeek', 'Month', 'Quarter', 'Year']].copy()
        combined_df = pd.concat([df_lags, future_df], ignore_index=True)
        combined_df.sort_values('Date', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Generate lag and rolling features
        for lag in [1, 7, 14, 30]:
            combined_df[f'{target_col}_lag_{lag}'] = combined_df[target_col].shift(lag)
        for window in [7, 14, 30]:
            combined_df[f'{target_col}_rolling_mean_{window}'] = combined_df[target_col].rolling(window=window).mean()
            combined_df[f'{target_col}_rolling_std_{window}'] = combined_df[target_col].rolling(window=window).std()

        # Seasonal features
        combined_df['sin_day'] = np.sin(2 * np.pi * combined_df['DayOfWeek'] / 7)
        combined_df['cos_day'] = np.cos(2 * np.pi * combined_df['DayOfWeek'] / 7)
        combined_df['sin_month'] = np.sin(2 * np.pi * combined_df['Month'] / 12)
        combined_df['cos_month'] = np.cos(2 * np.pi * combined_df['Month'] / 12)

        # Filter only future data
        future_df_full = combined_df[combined_df['Date'] > df['Date'].max()].copy()
        future_df_full.fillna(method='bfill', inplace=True)
        future_df_full.fillna(method='ffill', inplace=True)

        # Prepare future features
        future_features = [col for col in future_df_full.columns if col != 'Date']
        X_future = future_df_full[future_features]

        # Align with training features
        for col in X.columns:
            if col not in X_future.columns:
                X_future[col] = 0
        X_future = X_future[X.columns]

        # Predict
        if best_scaler:
            X_future_scaled = best_scaler.transform(X_future)
            future_predictions = best_model.predict(X_future_scaled)
        else:
            future_predictions = best_model.predict(X_future)

        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df[target_col], name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'{selected_target} Forecast (Next 30 Days)', xaxis_title='Date', yaxis_title=selected_target, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': np.round(future_predictions, 2)})
        st.markdown("### 📋 Forecast Details")
        st.dataframe(forecast_df, use_container_width=True)
