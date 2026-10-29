"""Response models (also the OpenAPI contract)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from brainct.constants import DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str


class ModelRef(BaseModel):
    name: str
    version: str
    head: str = Field(description='"seq" (series model) or "cnn" (lone slice)')


class Timing(BaseModel):
    decode_ms: float
    inference_ms: float
    total_ms: float


class Heatmap(BaseModel):
    slice_index: int
    label: str
    png_base64: str = Field(description="brain-window slice with a class activation overlay")


class SlicePrediction(BaseModel):
    index: int
    instance_number: int | None
    z: float | None
    probabilities: dict[str, float]
    positive: list[str] = Field(description="labels at or above their validation threshold")


class StudySummary(BaseModel):
    hemorrhage: bool = Field(description="study-level 'any' at the F1-optimal threshold")
    screening_positive: bool = Field(
        description="any slice above the 95%-sensitivity threshold (fewer misses, more alarms)"
    )
    probabilities: dict[str, float] = Field(description="max over slices, per label")
    subtypes: list[str]
    key_slice: int = Field(description="slice with the highest 'any' probability")


class SeriesResponse(BaseModel):
    model: ModelRef
    study: StudySummary
    slices: list[SlicePrediction]
    ordering: str
    warnings: list[str]
    heatmap: Heatmap | None = None
    timing: Timing
    disclaimer: str = DISCLAIMER


class SliceResponse(BaseModel):
    model: ModelRef
    prediction: SlicePrediction
    warnings: list[str]
    heatmap: Heatmap | None = None
    timing: Timing
    disclaimer: str = DISCLAIMER
