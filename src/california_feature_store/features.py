from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from datetime import timedelta

# Define the house entity - each house has a unique identifier
house = Entity(
    name="house",
    join_keys=["house_id"],
    description="A house in California with unique characteristics"
)

# Define the data source - parquet file with California Housing data
california_housing_source = FileSource(
    name="california_housing_source",
    path="data/california_housing.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Create FeatureView with the main features from California Housing dataset
california_housing_features = FeatureView(
    name="california_housing_features",
    entities=[house],
    ttl=timedelta(days=1),
    schema=[
        Field(name="median_income", dtype=Float64, description="Median income in the block group"),
        Field(name="house_age", dtype=Float64, description="Median age of houses in the block group"),
        Field(name="average_rooms", dtype=Float64, description="Average number of rooms per household"),
        Field(name="average_bedrooms", dtype=Float64, description="Average number of bedrooms per household"),
        Field(name="population", dtype=Float64, description="Population in the block group"),
        Field(name="average_occupants", dtype=Float64, description="Average number of occupants per household"),
        Field(name="latitude", dtype=Float64, description="Latitude of the block group"),
        Field(name="longitude", dtype=Float64, description="Longitude of the block group"),
        Field(name="median_house_value", dtype=Float64, description="Median house value in the block group (target variable)"),
    ],
    online=True,
    source=california_housing_source,
    tags={"team": "housing_analytics", "dataset": "california_housing"},
)
