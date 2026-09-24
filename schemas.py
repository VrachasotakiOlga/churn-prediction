from typing import Literal, Annotated

from pydantic import BaseModel, NonNegativeInt, PositiveFloat, Field
from pydantic_extra_types.country import CountryShortName

type UnitIntervalFloat = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


class UserData(BaseModel):
    user_id: NonNegativeInt
    days_since_registration: NonNegativeInt
    age_group: Literal["18-24", "25-34", "35-44", "45-54", "55+"]
    country: CountryShortName
    num_bets_last_30d: NonNegativeInt
    num_bets_last_7d: NonNegativeInt
    days_since_last_bet: NonNegativeInt
    num_deposits_30d: NonNegativeInt
    num_withdrawals_30d: NonNegativeInt
    favorite_sport: Literal["Football", "Basketball", "Tennis", "Horse_Racing", "Other"]


class ModelMetrics(BaseModel):
    accuracy: UnitIntervalFloat
    precision: UnitIntervalFloat
    recall: UnitIntervalFloat
    f1: UnitIntervalFloat
    roc_auc: UnitIntervalFloat


class Prediction(BaseModel):
    churn_probability: UnitIntervalFloat
