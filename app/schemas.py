from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    radius_mean: float = Field(..., ge=0, alias="radius_mean")
    texture_mean: float = Field(..., ge=0, alias="texture_mean")
    perimeter_mean: float = Field(..., ge=0, alias="perimeter_mean")
    area_mean: float = Field(..., ge=0, alias="area_mean")
    smoothness_mean: float = Field(..., ge=0, alias="smoothness_mean")
    compactness_mean: float = Field(..., ge=0, alias="compactness_mean")
    concavity_mean: float = Field(..., ge=0, alias="concavity_mean")
    concave_points_mean: float = Field(..., ge=0, alias="concave points_mean")
    symmetry_mean: float = Field(..., ge=0, alias="symmetry_mean")
    fractal_dimension_mean: float = Field(..., ge=0, alias="fractal_dimension_mean")

    radius_se: float = Field(..., ge=0, alias="radius_se")
    texture_se: float = Field(..., ge=0, alias="texture_se")
    perimeter_se: float = Field(..., ge=0, alias="perimeter_se")
    area_se: float = Field(..., ge=0, alias="area_se")
    smoothness_se: float = Field(..., ge=0, alias="smoothness_se")
    compactness_se: float = Field(..., ge=0, alias="compactness_se")
    concavity_se: float = Field(..., ge=0, alias="concavity_se")
    concave_points_se: float = Field(..., ge=0, alias="concave points_se")
    symmetry_se: float = Field(..., ge=0, alias="symmetry_se")
    fractal_dimension_se: float = Field(..., ge=0, alias="fractal_dimension_se")

    radius_worst: float = Field(..., ge=0, alias="radius_worst")
    texture_worst: float = Field(..., ge=0, alias="texture_worst")
    perimeter_worst: float = Field(..., ge=0, alias="perimeter_worst")
    area_worst: float = Field(..., ge=0, alias="area_worst")
    smoothness_worst: float = Field(..., ge=0, alias="smoothness_worst")
    compactness_worst: float = Field(..., ge=0, alias="compactness_worst")
    concavity_worst: float = Field(..., ge=0, alias="concavity_worst")
    concave_points_worst: float = Field(..., ge=0, alias="concave points_worst")
    symmetry_worst: float = Field(..., ge=0, alias="symmetry_worst")
    fractal_dimension_worst: float = Field(..., ge=0, alias="fractal_dimension_worst")
