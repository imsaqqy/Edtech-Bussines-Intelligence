# Full Year Business Forecast (2025)
# Import Essential Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.title("Full Year Business Forecast (2025)")

#Load Data
uploaded_file=st.file_uploader("Upload MIS Data Excel File",type=["xlsx"])
if uploaded_file:
    df=pd.read_excel(uploaded_file,sheet_name="Sheet1")
    df['Reg.Date']=pd.to_datetime(df['Reg.Date'],errors='coerce')
    df['YearMonth']=df['Reg.Date'].dt.to_period('M')
    #st.dataframe(df)
    #Filter Options   #unique()=no repeat
    technologies=df['Subject'] .dropna().unique()
    colleges=df['College'].dropna().unique()
    location=df['Location'].dropna().unique()
    selected_tech=st.selectbox("Select Technology",sorted(technologies))
    selected_colleges=st.selectbox("Select College (Optional)",['All']+sorted(colleges.tolist()))
    selected_locations=st.selectbox("Select Locations (Optional)",['All']+sorted(location.tolist()))
    #Apply filters
    data=df[df['Subject']==selected_tech]
    #st.dataframe(data)
    if selected_colleges!='All':
        data=data[data['College']==selected_colleges]
    if selected_locations!='All':
        data=data[data['Location']==selected_locations]
    #st.dataframe(data)
    #Group By Month
    monthly=data.groupby('YearMonth').size().reset_index(name='SNo.')
    monthly=monthly.set_index('YearMonth').asfreq('M').fillna(0)
    #st.dataframe(monthly)
    monthly.index=monthly.index.to_timestamp()
    if len(monthly)>=2:
        #prepare Regression model 
        X=np.array([d.toordinal() for d in monthly.index]).reshape(-1,1)#for dependent value

        #st.dataframe(X)
        y=monthly['SNo.'].values
        model=LinearRegression()
        model.fit(X,y)
        #predict all months of 2025
        future_dates=pd.date_range(start="2025-01-01", end="2025-12-01",freq='MS')
        X_future=np.array([d.toordinal() for d in future_dates]).reshape(-1,1)
        y_pred=model.predict(X_future)
        #st.dataframe(y_pred)

        #SHOW PREDICTED VALUES
        forecast_df=pd.DataFrame({
            'Month':future_dates.strftime('%B %Y'),
            'Predicted Enrollments':np.round(y_pred).astype(int)
        })
        st.subheader("Monthly Prediction for 2025")
        st.dataframe(forecast_df)
        #show total predictions
        st.success(f"Total Predicted Enrollments for 2025:{int(np.round(y_pred).sum())}")
    else:
        st.warning("Not enough data for prediction")