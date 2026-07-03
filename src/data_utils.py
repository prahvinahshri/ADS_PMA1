import pandas as pd

DATA_URL = ("https://raw.githubusercontent.com/wessamsw/"
            "Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv")

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    return pd.read_csv(url)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Arrival Delay'] = df['Arrival Delay'].fillna(df['Arrival Delay'].median())
    df = df[df['Cleanliness'].between(1, 5)].copy()
    return df
