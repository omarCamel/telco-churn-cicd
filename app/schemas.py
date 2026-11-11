from typing import Literal
from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    tenure: int = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
