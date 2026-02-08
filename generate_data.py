import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_records=1000):
    # Latitude and Longitude for HSR Layout, Bangalore
    # HSR Layout is roughly between 12.9080 and 12.9180 N, 77.6350 and 77.6450 E
    latitudes = np.random.uniform(12.9080, 12.9180, num_records)
    longitudes = np.random.uniform(77.6350, 77.6450, num_records)
    
    # Square footage: 500 to 5000
    total_sqft = np.random.randint(500, 5001, num_records)
    
    # Number of bedrooms based on sqft (rough mapping)
    # < 800: 1, 800-1500: 2, 1500-2500: 3, 2500-3500: 4, >3500: 5
    num_bedrooms = []
    for sqft in total_sqft:
        if sqft < 800:
            num_bedrooms.append(1)
        elif sqft < 1500:
            num_bedrooms.append(2)
        elif sqft < 2500:
            num_bedrooms.append(3)
        elif sqft < 3500:
            num_bedrooms.append(4)
        else:
            num_bedrooms.append(5)
    num_bedrooms = np.array(num_bedrooms)
    
    # Number of bathrooms: num_bedrooms to num_bedrooms + 1
    num_bathrooms = num_bedrooms + np.random.randint(0, 2, num_records)
    
    # Number of balconies: 0 to 3
    num_balconies = np.random.randint(0, 4, num_records)
    
    # House age: 0 to 30 years
    house_age = np.random.randint(0, 31, num_records)
    
    # Number of stories: 1 to 5
    num_stories = np.random.randint(1, 6, num_records)
    
    # Distance to MRT (Metro) in KM: 0.2 to 5.0
    distance_to_mrt = np.round(np.random.uniform(0.2, 5.0, num_records), 2)
    
    # Number of convenience stores: 0 to 12
    num_convenience_stores = np.random.randint(0, 13, num_records)
    
    # Gated community: 0 or 1
    is_gated_community = np.random.choice([0, 1], num_records, p=[0.4, 0.6])
    
    # Parking slots: 1 to 3
    parking_slots = np.random.randint(1, 4, num_records)
    
    # Property type: Apartment (0), Villa (1), Independent House (2)
    property_types = np.random.choice(['Apartment', 'Villa', 'Independent House'], num_records)
    
    # Floor number
    floor_numbers = []
    for p_type in property_types:
        if p_type == 'Apartment':
            floor_numbers.append(np.random.randint(0, 21))
        else:
            floor_numbers.append(np.random.randint(0, 3))
    floor_numbers = np.array(floor_numbers)
    
    # Distance to main road in meters
    distance_to_main_road = np.random.randint(10, 501, num_records)
    
    # Pricing Logic
    # Base price per sqft in HSR: 8000 to 15000
    base_price_per_sqft = np.random.randint(8000, 15001, num_records)
    
    # Adjustments:
    # 1. Age: -1% per year
    age_adjustment = 1 - (house_age * 0.01)
    
    # 2. Gated community: +15% premium
    gated_adjustment = np.where(is_gated_community == 1, 1.15, 1.0)
    
    # 3. Distance to MRT: -5% per km from 1km upwards
    mrt_adjustment = np.where(distance_to_mrt > 1.0, 1 - (distance_to_mrt - 1) * 0.05, 1.0)
    
    # 4. Convenience stores: +1% per store
    stores_adjustment = 1 + (num_convenience_stores * 0.01)
    
    # 5. Villa premium: +20%
    type_adjustment = np.where(property_types == 'Villa', 1.2, 1.0)
    
    # Calculate Final House Price in Lakhs
    # price = (sqft * base * adjustments) / 100,000
    # Adding some noise
    noise = np.random.normal(1, 0.05, num_records) # 5% variation
    
    house_price = (total_sqft * base_price_per_sqft * age_adjustment * gated_adjustment * mrt_adjustment * stores_adjustment * type_adjustment * noise) / 100000
    house_price = np.round(house_price, 2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'total_sqft': total_sqft,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'num_balconies': num_balconies,
        'num_stories': num_stories,
        'house_age': house_age,
        'distance_to_mrt': distance_to_mrt,
        'num_convenience_stores': num_convenience_stores,
        'latitude': latitudes,
        'longitude': longitudes,
        'is_gated_community': is_gated_community,
        'parking_slots': parking_slots,
        'property_type': property_types,
        'floor_number': floor_numbers,
        'distance_to_main_road': distance_to_main_road,
        'House Price': house_price
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(1000)
    output_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/hsr_house_prices.csv"
    df.to_csv(output_path, index=False)
    print(f"Success: Data generated and saved to {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Summary:")
    print(df.describe())
