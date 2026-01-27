from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Hotel Reservation Prediction API")

model = joblib.load("model/hotel_model.pkl")

# ---------- Pydantic Schema ----------
class HotelInput(BaseModel):

    no_of_adults: int = Field(..., ge=1, le=10)
    no_of_children: int = Field(..., ge=0, le=10)
    no_of_weekend_nights: int = Field(..., ge=0)
    no_of_week_nights: int = Field(..., ge=0)
    required_car_parking_space: int = Field(..., ge=0, le=1)
    lead_time: int = Field(..., ge=0)
    arrival_year: int = Field(..., ge=2017, le=2030)
    arrival_month: int = Field(..., ge=1, le=12)
    arrival_date: int = Field(..., ge=1, le=31)
    repeated_guest: int = Field(..., ge=0, le=1)
    no_of_previous_cancellations: int = Field(..., ge=0)
    no_of_previous_bookings_not_canceled: int = Field(..., ge=0)
    avg_price_per_room: float = Field(..., ge=0)
    no_of_special_requests: int = Field(..., ge=0)

    type_of_meal_plan: str
    room_type_reserved: str
    market_segment_type: str


# ---------- Prediction ----------
@app.post("/predict")
def predict(data: HotelInput):
    try:
        df = pd.DataFrame([data.dict()])
        pred = model.predict(df)[0]
        result = "Canceled" if pred == 1 else "Not Canceled"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
