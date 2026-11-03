"""Request and response models.

Validation lives here rather than in the routers, so a bad payload is
rejected with a 422 before any database or NumPy work happens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if v.isdigit():
            raise ValueError("Password cannot be all digits.")
        if v.lower() in {"password", "12345678", "qwertyui"}:
            raise ValueError("That password is too common.")
        return v

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be blank.")
        return v.strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    name: str
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    filename: str
    n_rows: int
    n_columns: int
    created_at: datetime


class DatasetDetail(DatasetOut):
    columns: list[str] = Field(default_factory=list)
    preview: list[dict[str, str]] = Field(default_factory=list)
    suggested_target: str | None = None
    suggested_task: str | None = None


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


class TrainRequest(BaseModel):
    dataset_id: str
    target: str = Field(min_length=1, max_length=200)
    task: Literal["classification", "regression"]
    test_fraction: float = Field(default=0.25, gt=0.05, lt=0.6)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    feature_columns: list[str] | None = None


class CandidateOut(BaseModel):
    name: str
    family: str
    params: dict
    scores: dict[str, float]
    train_seconds: float
    error: str | None = None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    target: str
    task: str
    status: str
    winner_name: str | None
    winner_family: str | None
    primary_metric: str
    primary_score: float | None
    baseline_score: float | None
    n_rows: int
    n_features: int
    train_rows: int
    test_rows: int
    duration_seconds: float
    created_at: datetime


class RunDetail(RunOut):
    leaderboard: list[CandidateOut] = Field(default_factory=list)
    baseline: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """One row to score, as a mapping of source column -> value."""

    row: dict[str, str | float | int | None]


class PredictResponse(BaseModel):
    prediction: str | float
    task: str
    model: str
    # For classification: probability per class, when the model exposes them.
    probabilities: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
