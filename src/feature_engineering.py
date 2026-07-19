import numpy as np
import pandas as pd

def add_time_features(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    data["hour"] = data["pickup_datetime"].dt.hour
    data["month"] = data["pickup_datetime"].dt.month
    data["day_of_month"] = data["pickup_datetime"].dt.day.astype(int)
    data["day_of_week"] = data["pickup_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encoding
    hour_rad = 2 * np.pi * data["hour"] / 24
    data["hour_sin"] = np.sin(hour_rad)
    data["hour_cos"] = np.cos(hour_rad)

    # Rush hour (5 PM - 10 PM)
    # Busy hour (8 AM - 6 PM)
    data["rush_hour"] = data["hour"].between(17, 22).astype(int)
    data["busy_hour"] = data["hour"].between(8, 18).astype(int)

    # cross feature
    data['week_day_hour'] = data['hour'] * data['day_of_week']


def add_distance_features(data):
    R = 6371.0  # Earth's radius (km)

    lat1 = np.radians(data["pickup_latitude"])
    lon1 = np.radians(data["pickup_longitude"])
    lat2 = np.radians(data["dropoff_latitude"])
    lon2 = np.radians(data["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    def add_haversine_distance():
        a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        data["distance_haversine"] = R * c


    def add_manhattan_distance():
        lat_distance = R * np.abs(dlat)
        lon_distance = R * np.cos((lat1 + lat2) / 2) * np.abs(dlon)

        data["distance_manhattan"] = lat_distance + lon_distance

    def add_euclidean_distance():
        data["distance_euclidean"] = np.sqrt(
            (data["dropoff_longitude"] - data["pickup_longitude"]) ** 2 +
            (data["dropoff_latitude"] - data["pickup_latitude"]) ** 2
        )

    def add_bearing_features():
        y = np.sin(dlon) * np.cos(lat2)
        x = (
                np.cos(lat1) * np.sin(lat2)
                - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )

        bearing = np.degrees(np.arctan2(y, x))
        bearing_rad = np.radians(bearing)

        data["bearing_sin"] = np.sin(bearing_rad)
        data["bearing_cos"] = np.cos(bearing_rad)

    add_haversine_distance()
    add_manhattan_distance()
    add_euclidean_distance()
    add_bearing_features()


def apply_log_transformation(data):
    cols = [
        "distance_haversine",
        "distance_manhattan",
        "distance_euclidean",
    ]

    for col in cols:
        if col in data.columns:
            data[f"log_{col}"] = np.log1p(data[col])


def build_features(data, kmeans):
    data["pickup_cluster"] = kmeans.predict(
        data[["pickup_longitude", "pickup_latitude"]].values
    )
    data["dropoff_cluster"] = kmeans.predict(
        data[["dropoff_longitude", "dropoff_latitude"]].values
    )

    add_time_features(data)
    add_distance_features(data)
    apply_log_transformation(data)