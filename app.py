import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --- Load and melt the data ---
@st.cache_data
def load_and_prepare_data():
    # Load datasets
    confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

    confirmed = pd.read_csv(confirmed_url)
    deaths = pd.read_csv(deaths_url)

    def melt_df(df, value_name):
        df_melted = df.drop(columns=["Lat", "Long"]).melt(
            id_vars=["Province/State", "Country/Region"],
            var_name="Date", value_name=value_name
        )
        df_melted["Date"] = pd.to_datetime(df_melted["Date"], format="%m/%d/%y")
        return df_melted

    confirmed_long = melt_df(confirmed, "confirmed")
    deaths_long = melt_df(deaths, "deaths")
    return confirmed_long, deaths_long

confirmed_long, deaths_long = load_and_prepare_data()

# --- Sidebar controls ---0.'00000000000

st.sidebar.header("Dashboard Controls")

# Country selector
countries = sorted(confirmed_long["Country/Region"].unique())
selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("India"))

# Forecast horizon
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=60, value=30, step=1)

# Date range filter
min_date = confirmed_long["Date"].min()
max_date = confirmed_long["Date"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# --- Filter data by country and date ---
country_data = confirmed_long[confirmed_long["Country/Region"] == selected_country]
country_deaths = deaths_long[deaths_long["Country/Region"] == selected_country]

confirmed_series = country_data.groupby("Date")["confirmed"].sum().loc[date_range[0]:date_range[1]]
deaths_series = country_deaths.groupby("Date")["deaths"].sum().loc[date_range[0]:date_range[1]]

# Ensure daily frequency
confirmed_series = confirmed_series.asfreq("D").fillna(method="ffill")
deaths_series = deaths_series.asfreq("D").fillna(method="ffill")

# --- Modeling and Forecasting ---
model = ARIMA(confirmed_series, order=(5, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_days)

# Anomaly detection
daily_cases = confirmed_series.diff().fillna(0)
z_scores = (daily_cases - daily_cases.mean()) / daily_cases.std()
anomalies = daily_cases[z_scores > 2]

# --- Dashboard UI ---
st.title("🦠 COVID-19 Disease Surveillance Dashboard")

# Latest values
st.subheader(f"📍 {selected_country} COVID-19 Stats")
st.write(f"Total Confirmed Cases: **{int(confirmed_series.iloc[-1])}**")
st.write(f"Total Deaths: **{int(deaths_series.iloc[-1])}**")

# Time Series + Forecast Plot
st.subheader(f"📊 Confirmed Cases + Forecast ({forecast_days} days)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(confirmed_series.index, confirmed_series.values, label="Confirmed Cases")
ax.plot(forecast.index, forecast.values, color="red", label="Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Cases")
ax.legend()
st.pyplot(fig)

# Anomaly Detection Plot
st.subheader("⚠️ Daily New Cases with Anomalies")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(daily_cases.index, daily_cases.values, label="Daily Cases")
ax2.scatter(anomalies.index, anomalies.values, color="red", label="Anomalies")
ax2.set_xlabel("Date")
ax2.set_ylabel("New Cases")
ax2.legend()
st.pyplot(fig2)

# Forecast Table
st.subheader("📅 Forecast Table")
forecast_df = pd.DataFrame({"Date": forecast.index, "Forecasted Cases": forecast.values.astype(int)})
st.dataframe(forecast_df.set_index("Date"))
