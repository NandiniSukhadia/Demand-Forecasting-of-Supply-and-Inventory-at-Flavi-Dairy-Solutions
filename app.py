def create_forecasting_section(df):
    """Create the forecasting section"""
    st.markdown("## 🔮 Demand Forecasting")

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

    # Validate enough data
    if df.shape[0] < 60:
        st.error("Not enough data to generate lag/rolling features. Please use at least 60 days of historical data.")
        return

    # Prepare features
    X, y, df_processed = prepare_features(df.copy(), target_col)

    if st.button("🚀 Train Forecasting Models"):
        with st.spinner("Training models..."):
            try:
                results, X_train, X_test, y_train, y_test = train_models(X, y)
            except Exception as e:
                st.error(f"Error training models: {e}")
                return

        st.markdown("### 📊 Model Performance Comparison")

        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'R²': result['r2']
            })

        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]['model']
        best_scaler = results[best_model_name]['scaler']

        st.success(f"🎯 Best Model: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.2f})")

        # Future forecast
        st.markdown("### 🔮 Future Forecast")

        try:
            future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
            future_df['Month'] = future_df['Date'].dt.month
            future_df['Quarter'] = future_df['Date'].dt.quarter
            future_df['Year'] = future_df['Date'].dt.year

            # Combine with historical data
            df_lags = df[['Date', target_col, 'DayOfWeek', 'Month', 'Quarter', 'Year']].copy()
            combined = pd.concat([df_lags, future_df], ignore_index=True)
            combined.sort_values('Date', inplace=True)
            combined.reset_index(drop=True, inplace=True)

            # Lag and rolling features
            for lag in [1, 7, 14, 30]:
                combined[f'{target_col}_lag_{lag}'] = combined[target_col].shift(lag)
            for window in [7, 14, 30]:
                combined[f'{target_col}_rolling_mean_{window}'] = combined[target_col].rolling(window=window).mean()
                combined[f'{target_col}_rolling_std_{window}'] = combined[target_col].rolling(window=window).std()

            combined['sin_day'] = np.sin(2 * np.pi * combined['DayOfWeek'] / 7)
            combined['cos_day'] = np.cos(2 * np.pi * combined['DayOfWeek'] / 7)
            combined['sin_month'] = np.sin(2 * np.pi * combined['Month'] / 12)
            combined['cos_month'] = np.cos(2 * np.pi * combined['Month'] / 12)

            future_df_full = combined[combined['Date'] > df['Date'].max()].copy()
            future_df_full.fillna(method='bfill', inplace=True)
            future_df_full.fillna(method='ffill', inplace=True)

            # Match features
            future_features = [col for col in future_df_full.columns if col != 'Date']
            X_future = future_df_full[future_features]
            for col in X.columns:
                if col not in X_future.columns:
                    X_future[col] = 0
            X_future = X_future[X.columns]

            # Predict
            if best_scaler:
                X_future = best_scaler.transform(X_future)
            future_preds = best_model.predict(X_future)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df[target_col], name='Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Forecast', line=dict(color='red', dash='dash')))
            fig.update_layout(title=f"{selected_target} Forecast", xaxis_title='Date', yaxis_title=selected_target, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': np.round(future_preds, 2)})
            st.markdown("### 📋 Forecast Details")
            st.dataframe(forecast_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error during forecasting: {e}")
