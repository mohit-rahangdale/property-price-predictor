import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data():
    """Load and preprocess dataset"""
    df = pd.read_csv('data/Bengaluru_House_Data.csv')
    
    # Data cleaning
    df = df.dropna(subset=['size', 'bath', 'price'])
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    
    df['bhk'] = df['size'].str.split().str.get(0).astype(float)
    df = df[df['bhk'] <= 20]  
    
    def convert_sqft(x):
        if '-' in str(x):
            vals = str(x).split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        try:
            return float(x)
        except:
            return np.nan
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft'])
    
    df['area_type_original'] = df['area_type']
    df['location_original'] = df['location']
    
    top_locations = df['location'].value_counts().head(20).index
    df['location'] = df['location'].apply(
        lambda x: x if x in top_locations else 'Other'
    )
    
    le = LabelEncoder()
    df['area_type'] = le.fit_transform(df['area_type'])
    df['location'] = le.fit_transform(df['location'])
    
    features = ['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk']
    target = 'price'
    
    return df, features, target, top_locations

def train_model():
    """Train and save Random Forest model"""
    df, features, target, top_locations = load_data()
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    # Save model
    joblib.dump(model, 'models/house_price_model.pkl')
    print("Model saved to models/house_price_model.pkl")

    # Save feature metadata for Streamlit
    metadata = {
        'area_types': df['area_type_original'].unique().tolist(),
        'top_locations': top_locations.tolist()
    }
    joblib.dump(metadata, 'models/feature_metadata.pkl')
    print("Feature metadata saved")

if __name__ == "__main__":
    train_model()