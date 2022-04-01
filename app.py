
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from keras.models import load_model
import plotly.graph_objects as go
import time




FullData = pickle.load(open("FullData.pkl", "rb"))
FullData.reset_index(inplace = True)

file = open("StockData.pkl", "rb")
data = pickle.load(file)    

model = load_model("LSTM_model.h5")

st.title('Stock Price Prediction Application')
st.caption('In today’s world companies and businesses are very keen to understand and analyze the market to lower their expenses and enhance profits. With modern data analytics, we can identify purchasing and selling patterns by carefully analyzing the data to help the investors. Here, We are tring to predict future value of Adani Stock.')
st.image('Adani_2012_logo.png')

input = st.text_input('Enter Days For Predict Future Values' ,placeholder='Days')
st.write('You Have Entered ', input, ' Days')


input = int(input)

if type(input) == int:
    st.subheader('All Information About Stock')
    st.table(FullData.describe())

    st.subheader('Stock Graph From February 2011 to Decemebr 2021   ')

    trace1 = go.Scatter(
        x = FullData['Date'],
        y = FullData['Close'],
        mode = 'lines',
        name = 'Actual Price'
    )
    layout = go.Layout(
        title = "Adani Enterprise Stock",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )

    fig = go.Figure(data=trace1, layout=layout)
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

    #Function
    look_back = 10
    def predict(input, model):
        prediction_list = data[-look_back:]
        
        for _ in range(input):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list
        
    def predict_dates(input):
        last_date = FullData['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=input+1).tolist()
        return prediction_dates

    

    forecast = predict(input, model)
    forecast_dates = predict_dates(input)

    st.subheader(f'Prediction Chart For {input} Days')

    with st.spinner('Wait for it...'):
        time.sleep(5)

        trace2 = go.Scatter(
            x = forecast_dates,
            y = forecast,
            mode = 'lines',
            name = 'Prediction',
            line_color="#04eb00"
        )
        layout = go.Layout(
            title = "Adani Enterprise Stock",
            xaxis = {'title' : "Date"},
            yaxis = {'title' : "Close"}
        )   

        fig = go.Figure(data=trace2, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    
else: 
    st.caption('Please A Number ', unsafe_allow_html=False)
























