import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

df = pd.read_excel('datasets/penny_part_2/fed_diary_of_consumer_payment_choice/2025-diary-of-consumer-payment-choice-charts.xlsx', sheet_name='Figure 4')
df.rename(columns={'Unnamed: 0': 'Years'}, inplace=True)
df['Years'] = pd.to_datetime(df['Years'], format='%Y')
months = df.loc[:len(df)-2, ['Years','Cash']].copy()
df['Cash'] = df['Cash']*100

# trying lin reg for more data points - not useful
months_list = []
for i in range(len(df) - 1):
    slope = (df.loc[i + 1, 'Cash'] - df.loc[i, 'Cash']) / 12
    y_int = df.loc[i, 'Cash']
    for n in range(12):
        current_date = df.loc[i, 'Years'] + pd.DateOffset(months=n)
        current_cash = y_int + slope * n
        months_list.append({'Date': current_date, 'Cash': current_cash})
months_df = pd.DataFrame(months_list)
cash_df = months_df[['Date','Cash']]
print(months_df)
print(cash_df)

p,d,q = 2,2,1
model = SARIMAX(cash_df['Cash'], seasonal_order=(p,d,q,12))
model_fit = model.fit()
prediction = model_fit.forecast(steps=len(cash_df))
mse = mean_squared_error(cash_df['Cash'], prediction)

forecast = model_fit.forecast(steps=48)
last_year = cash_df['Date'].dt.year.iloc[-1]
future_years = pd.date_range(start=f'{last_year+1}', periods=48, freq='ME')

# Combine for plotting
forecast_df = pd.DataFrame({'Cash': forecast.values}, index=future_years)

plt.plot(cash_df['Date'], cash_df['Cash'], label="Known Values")
plt.plot(forecast_df.index, forecast_df['Cash'], label="Forecast", linestyle='--', color='red')
plt.xlabel("Date")
plt.ylabel("Transactions in Cash (%)")
plt.legend()
plt.title("ARIMA Prediction")
plt.show()