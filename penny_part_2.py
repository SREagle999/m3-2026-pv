import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

df = pd.read_excel('datasets/penny_part_2/fed_diary_of_consumer_payment_choice/2025-diary-of-consumer-payment-choice-charts.xlsx', sheet_name='Figure 4')
df.rename(columns={'Unnamed: 0': 'Years'}, inplace=True)
df['Years'] = pd.to_datetime(df['Years'], format='%Y')
df['Cash'] = df['Cash']*100
cash_df = df[['Years','Cash']]

p,d,q = 0,0,0
best_mse = 10000
best_set = (p,d,q)
for p in range(0,3):
    for d in range(0,3):
        for q in range(0,3):
            model = SARIMAX(df['Cash'], order=(p,d,q))
            model_fit = model.fit()
            prediction = model_fit.forecast(steps=len(cash_df))
            mse = mean_squared_error(cash_df['Cash'], prediction)
            if mse < best_mse:
                best_mse = mse
                best_set = (p,d,q)
                best_model = model
                best_prediction = prediction
print(best_set)

plt.plot(cash_df['Years'], cash_df['Cash'], label="Known Values")
plt.xlabel("Year")
plt.ylabel("Transactions in Cash (%)")
plt.xticks(rotation=45)

# Grid/Auto-Manual Fit ARIMA w/ box-cox
plt.legend()
plt.title("ARIMA Prediction")
plt.show()