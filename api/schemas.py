from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: float = Field(ge=0)
    product_type: Literal["L", "M", "H"]


class PredictionResponse(BaseModel):
    risk_score: float
    failure_predicted: bool
    decision_threshold: float