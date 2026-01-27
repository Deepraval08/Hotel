import streamlit as st
import requests

st.title("🏨 Hotel Reservation Cancellation Predictor")

st.subheader("Enter Booking Details")

# -------- Inputs --------
no_of_adults = st.number_input("Adults", 1, 10, 1)
no_of_children = st.number_input("Children", 0, 10, 0)
no_of_weekend_nights = st.number_input("Weekend Nights", 0, 10, 0)
no_of_week_nights = st.number_input("Week Nights", 0, 20, 1)
required_car_parking_space = st.selectbox("Parking Required", [0,1])
lead_time = st.number_input("Lead Time", 0)
arrival_year = st.number_input("Arrival Year", 2017, 2030, 2025)
arrival_month = st.selectbox("Arrival Month", list(range(1,13)))
arrival_date = st.selectbox("Arrival Date", list(range(1,32)))
repeated_guest = st.selectbox("Repeated Guest", [0,1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", 0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Successful Bookings", 0)
avg_price_per_room = st.number_input("Average Price", 0.0)
no_of_special_requests = st.number_input("Special Requests", 0)

type_of_meal_plan = st.selectbox(
    "Meal Plan", 
    ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"]
)

room_type_reserved = st.selectbox(
    "Room Type",
    ["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4"]
)

market_segment_type = st.selectbox(
    "Market Segment",
    ["Online","Offline","Corporate","Aviation","Complementary"]
)

# -------- Submit --------
if st.button("Predict"):

    payload = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "required_car_parking_space": required_car_parking_space,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
            "type_of_meal_plan": type_of_meal_plan,
            "room_type_reserved": room_type_reserved,
            "market_segment_type": market_segment_type
        }

    with st.spinner("Predicting... Please wait"):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
        )


    if response.status_code == 200:
        st.success(f"Prediction: {response.json()['prediction']}")
    else:
        st.error("API Error")

