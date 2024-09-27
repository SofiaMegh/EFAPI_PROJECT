from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import psutil
import logging

# Load the trained model
model = joblib.load('energy_model.pkl')

# Create a FastAPI instance
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the input data model
class EnergyInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    ram_usage: float = Field(..., ge=0, le=100, description="RAM usage percentage")
    disk_activity: float = Field(..., ge=0, le=100, description="Disk activity percentage")
    battery_usage: float = Field(..., ge=0, le=100, description="Battery usage percentage")

@app.post("/predict/")
def predict(input_data: EnergyInput):
    features = [[input_data.cpu_usage, input_data.ram_usage, input_data.disk_activity, input_data.battery_usage]]
    prediction = model.predict(features)
    logger.info(f"Prediction made with input: {input_data}, result: {prediction[0]}")
    return {"predicted_energy_generated": prediction[0]}

@app.get("/predict/real-time/")
def predict_real_time():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        disk_activity = psutil.disk_usage('/').percent  # Modify '/' to the relevant path if needed
        battery = psutil.sensors_battery()
        battery_usage = battery.percent if battery else 0
        
        features = [[cpu_usage, ram_usage, disk_activity, battery_usage]]
        prediction = model.predict(features)
        logger.info(f"Real-time prediction made: CPU={cpu_usage}, RAM={ram_usage}, Disk={disk_activity}, Battery={battery_usage}, Result={prediction[0]}")
        
        return {"predicted_energy_generated": prediction[0]}
    except Exception as e:
        logger.error(f"Error in real-time prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching real-time data.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

