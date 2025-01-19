from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

# Initialize FastAPI app
app = FastAPI(title="Maudlin API", description="API for interacting with Maudlin framework", version="1.0.0")

# Define base directories
MAUDLIN_DATA_DIR = os.getenv("MAUDLIN_DATA_DIR", "./maudlin_data")

# Data models
class UnitSelection(BaseModel):
    unit_name: str

class TrainingRequest(BaseModel):
    unit_name: str

class PredictionRequest(BaseModel):
    unit_name: str
    input_file: str  # Path to the prediction CSV file

# Helper function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {result.stderr}")
    return result.stdout

# Endpoints

@app.get("/units", summary="List all available units")
def list_units():
    try:
        output = run_command("mdln list")
        return {"units": output.strip().split("\n")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/units/select", summary="Select a unit")
def select_unit(unit: UnitSelection):
    try:
        run_command(f"mdln use {unit.unit_name}")
        return {"message": f"Unit {unit.unit_name} selected."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", summary="Run training for a unit")
def train_unit(request: TrainingRequest):
    try:
        output = run_command(f"mdln train {request.unit_name}")
        return {"message": f"Training started for unit {request.unit_name}.", "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", summary="Run predictions for a unit")
def predict_unit(request: PredictionRequest):
    try:
        output = run_command(f"mdln predict {request.unit_name} --prediction_csv_path={request.input_file}")
        return {"message": f"Prediction completed for unit {request.unit_name}.", "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{run_id}", summary="Fetch reports for a specific run")
def get_report(run_id: str):
    report_dir = os.path.join(MAUDLIN_DATA_DIR, "output", run_id)
    if not os.path.exists(report_dir):
        raise HTTPException(status_code=404, detail="Report not found.")
    # Optionally, return a list of files or serve specific reports
    files = os.listdir(report_dir)
    return {"run_id": run_id, "reports": files}
