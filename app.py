from flask import Flask,request,jsonify,render_template
import numpy
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import yfinance as yf
from datetime import date,timedelta 
import pandas_ta as ta 

model = keras.models.load_model("lala_trading_model.h5", compile=False)

# Compile the model again with the correct loss function
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

app = Flask(__name__)

time_step = 150
index_of_close = 2
no_of_features = 3


@app.route('/')
def home():

    yesterday = date.today()
    day_before_yesterday = yesterday-timedelta(days=1)
    prev_days = yesterday-timedelta(days=400)
    # data = yf.download("MSFT", start = prev_days, end = day_before_yesterday, interval="1d")
    data = yf.download("MSFT", start = prev_days, end = yesterday, interval="1d")
    data.columns = data.columns.droplevel(1)
    data = data[-time_step:]

    scaler = MinMaxScaler(feature_range=(0,1))

    feature_cols = ['High', 'Low', 'Close']
    data = scaler.fit_transform(data[feature_cols])

    data = data.reshape(1,data.shape[0],data.shape[1])

    y_hat = model.predict(data)

    dummy = numpy.zeros((y_hat.shape[0],no_of_features))
    dummy[:,index_of_close] = y_hat
    y_hat = scaler.inverse_transform(dummy)[:,index_of_close]
    print(f'Prediction for {yesterday} is ${y_hat[0]}')
    
    return render_template('index.html', prediction=round(y_hat[0], 2), date=yesterday)

if __name__ == '__main__':
    app.run(debug=True)