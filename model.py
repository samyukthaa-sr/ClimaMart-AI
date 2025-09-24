import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

data = pd.read_csv('data.csv') 

le_location = LabelEncoder()
le_weather = LabelEncoder()
le_category = LabelEncoder()
le_product = LabelEncoder()
le_demand = LabelEncoder()

data['Location_encoded'] = le_location.fit_transform(data['Location'])
data['Weather_encoded'] = le_weather.fit_transform(data['Weather'])
data['Category_encoded'] = le_category.fit_transform(data['Category'])
data['Product_encoded'] = le_product.fit_transform(data['Product'])
data['Demand_encoded'] = le_demand.fit_transform(data['Demand Level'])

high_demand_data = data[data['Demand Level'] == 'High'].copy()

X_product = high_demand_data[['Location_encoded', 'Weather_encoded']]
y_product = high_demand_data['Product_encoded']

X_train_prod, X_test_prod, y_train_prod, y_test_prod = train_test_split(
    X_product, y_product, test_size=0.2, random_state=42
)

product_model = RandomForestClassifier(n_estimators=100, random_state=42)
product_model.fit(X_train_prod, y_train_prod)

y_pred_prod = product_model.predict(X_test_prod)
product_accuracy = accuracy_score(y_test_prod, y_pred_prod)
print(f"Product Prediction Accuracy: {product_accuracy:.2%}")

X_price = high_demand_data[['Location_encoded', 'Weather_encoded', 'Product_encoded']]
y_price = high_demand_data[['Suggested Price Min (₹/kg)', 'Suggested Price Max (₹/kg)']]

X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)

price_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
price_model.fit(X_train_price, y_train_price)

y_pred_price = price_model.predict(X_test_price)
min_mae = mean_absolute_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0])
max_mae = mean_absolute_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1])
min_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0]))
max_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1]))

print("\nPrice Prediction Evaluation:")
print(f"Min Price - MAE: {min_mae:.2f}, RMSE: {min_rmse:.2f}")
print(f"Max Price - MAE: {max_mae:.2f}, RMSE: {max_rmse:.2f}")

def predict_products_and_prices(location, weather):
    try:
        loc_encoded = le_location.transform([location])[0]
        weather_encoded = le_weather.transform([weather])[0]
    except ValueError as e:
        return f"Error: {e}. Please check your inputs."
    
    product_input = pd.DataFrame([[loc_encoded, weather_encoded]], 
                                columns=['Location_encoded', 'Weather_encoded'])
    
    product_encoded = product_model.predict(product_input)[0]
    high_demand_product = le_product.inverse_transform([product_encoded])[0]
    
    price_input = pd.DataFrame([[loc_encoded, weather_encoded, product_encoded]], 
                              columns=['Location_encoded', 'Weather_encoded', 'Product_encoded'])
    
    price = price_model.predict(price_input)[0]
    high_min_price, high_max_price = price[0], price[1]
    
    medium_demand_data = data[(data['Location'] == location) & 
                             (data['Weather'] == weather) & 
                             (data['Demand Level'] == 'Medium')]
    
    medium_products = []
    if not medium_demand_data.empty:
        medium_grouped = medium_demand_data.groupby('Product').agg({
            'Suggested Price Min (₹/kg)': 'mean',
            'Suggested Price Max (₹/kg)': 'mean'
        }).reset_index()
        
        for _, row in medium_grouped.iterrows():
            medium_products.append({
                'Product': row['Product'],
                'Price Range': f"₹{row['Suggested Price Min (₹/kg)']:.1f} - ₹{row['Suggested Price Max (₹/kg)']:.1f} per kg"
            })
    
    return {
        'Location': location,
        'Weather': weather,
        'High Demand Product': high_demand_product,
        'High Demand Price Range': f"₹{high_min_price:.1f} - ₹{high_max_price:.1f} per kg",
        'Medium Demand Products': medium_products
    }

available_locations = sorted(data['Location'].unique())
available_weather = sorted(data['Weather'].unique())

def get_valid_input(prompt, options, option_type):
    while True:
        print(f"\nAvailable {option_type}s:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        user_input = input(f"\nEnter {option_type} (name or number): ").strip()
        
        if user_input.isdigit():
            index = int(user_input) - 1
            if 0 <= index < len(options):
                return options[index]
        
        if user_input in options:
            return user_input
        
        print(f"Invalid {option_type}. Please try again.")

location_input = get_valid_input("Enter location", available_locations, "location")

weather_input = get_valid_input("Enter weather condition", available_weather, "weather")

prediction = predict_products_and_prices(location_input, weather_input)

print("\n" + "="*50)
print(f"PREDICTION RESULTS FOR {prediction['Location'].upper()}")
print("="*50)
print(f"Weather Condition: {prediction['Weather']}")
print("\nHIGH DEMAND PRODUCT:")
print(f"  Product: {prediction['High Demand Product']}")
print(f"  Suggested Price Range: {prediction['High Demand Price Range']}")

print("\nMEDIUM DEMAND PRODUCTS:")
if prediction['Medium Demand Products']:
    for product in prediction['Medium Demand Products']:
        print(f"  Product: {product['Product']}")
        print(f"  Suggested Price Range: {product['Price Range']}")
else:
    print("  No medium-demand products found for this location and weather condition.")

print("\n" + "="*50)






'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# Load the dataset
data = pd.read_csv('data.csv') 

# Preprocessing
# Encode categorical variables
le_location = LabelEncoder()
le_weather = LabelEncoder()
le_category = LabelEncoder()
le_product = LabelEncoder()
le_demand = LabelEncoder()

data['Location_encoded'] = le_location.fit_transform(data['Location'])
data['Weather_encoded'] = le_weather.fit_transform(data['Weather'])
data['Category_encoded'] = le_category.fit_transform(data['Category'])
data['Product_encoded'] = le_product.fit_transform(data['Product'])
data['Demand_encoded'] = le_demand.fit_transform(data['Demand Level'])

# Filter high-demand products
high_demand_data = data[data['Demand Level'] == 'High'].copy()

# Prepare features and targets for product prediction
X_product = high_demand_data[['Location_encoded', 'Weather_encoded']]
y_product = high_demand_data['Product_encoded']

# Split data for product prediction
X_train_prod, X_test_prod, y_train_prod, y_test_prod = train_test_split(
    X_product, y_product, test_size=0.2, random_state=42
)

# Train Random Forest Classifier for product prediction
product_model = RandomForestClassifier(n_estimators=100, random_state=42)
product_model.fit(X_train_prod, y_train_prod)

# Evaluate product prediction
y_pred_prod = product_model.predict(X_test_prod)
product_accuracy = accuracy_score(y_test_prod, y_pred_prod)
print(f"Product Prediction Accuracy: {product_accuracy:.2%}")

# Prepare features and targets for price prediction
X_price = high_demand_data[['Location_encoded', 'Weather_encoded', 'Product_encoded']]
y_price = high_demand_data[['Suggested Price Min (₹/kg)', 'Suggested Price Max (₹/kg)']]

# Split data for price prediction
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)

# Train Random Forest Regressor for price prediction
price_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
price_model.fit(X_train_price, y_train_price)

# Evaluate price prediction
y_pred_price = price_model.predict(X_test_price)
min_mae = mean_absolute_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0])
max_mae = mean_absolute_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1])
min_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0]))
max_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1]))

print("\nPrice Prediction Evaluation:")
print(f"Min Price - MAE: {min_mae:.2f}, RMSE: {min_rmse:.2f}")
print(f"Max Price - MAE: {max_mae:.2f}, RMSE: {max_rmse:.2f}")

# Prediction function
def predict_product_and_price(location, weather):
    # Encode inputs
    try:
        loc_encoded = le_location.transform([location])[0]
        weather_encoded = le_weather.transform([weather])[0]
    except ValueError as e:
        return f"Error: {e}. Please check your inputs."
    
    # Create DataFrames with proper column names
    product_input = pd.DataFrame([[loc_encoded, weather_encoded]], 
                                columns=['Location_encoded', 'Weather_encoded'])
    
    # Predict product
    product_encoded = product_model.predict(product_input)[0]
    product = le_product.inverse_transform([product_encoded])[0]
    
    # Create input for price prediction
    price_input = pd.DataFrame([[loc_encoded, weather_encoded, product_encoded]], 
                              columns=['Location_encoded', 'Weather_encoded', 'Product_encoded'])
    
    # Predict price
    price = price_model.predict(price_input)[0]
    min_price, max_price = price[0], price[1]
    
    return {
        'Location': location,
        'Weather': weather,
        'High Demand Product': product,
        'Suggested Price Range': f"₹{min_price:.1f} - ₹{max_price:.1f} per kg"
    }

# Example usage
print("\nPrediction:")
print(predict_product_and_price('Chennai', 'Mist'))
print(predict_product_and_price('Coimbatore', 'Cloudy, chance of rain'))
'''



'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('data.csv') 

# Data Exploration
print("Dataset shape:", data.shape)
print("\nDemand Level Distribution:")
print(data['Demand Level'].value_counts())

# Check high-demand products count
high_demand_data = data[data['Demand Level'] == 'High'].copy()
print("\nNumber of high-demand products:", len(high_demand_data))
print("Unique high-demand products:", high_demand_data['Product'].nunique())
print("\nHigh-demand product distribution:")
print(high_demand_data['Product'].value_counts())

# Preprocessing
# Encode categorical variables
le_location = LabelEncoder()
le_weather = LabelEncoder()
le_category = LabelEncoder()
le_product = LabelEncoder()
le_demand = LabelEncoder()

data['Location_encoded'] = le_location.fit_transform(data['Location'])
data['Weather_encoded'] = le_weather.fit_transform(data['Weather'])
data['Category_encoded'] = le_category.fit_transform(data['Category'])
data['Product_encoded'] = le_product.fit_transform(data['Product'])
data['Demand_encoded'] = le_demand.fit_transform(data['Demand Level'])

# Feature Engineering
# Create interaction features
data['Temp_Humidity_Interaction'] = data['Temperature(°C)'] * data['Humidity(%)']
data['Location_Weather_Interaction'] = data['Location_encoded'] * data['Weather_encoded']

# Create price range feature
data['Price_Range'] = data['Suggested Price Max (₹/kg)'] - data['Suggested Price Min (₹/kg)']

# Filter high-demand products
high_demand_data = data[data['Demand Level'] == 'High'].copy()

# Check if we have enough samples for each product
product_counts = high_demand_data['Product'].value_counts()
min_samples_per_product = product_counts.min()
print(f"\nMinimum samples per high-demand product: {min_samples_per_product}")

# If we have too few samples for some products, we'll use upsampling
if min_samples_per_product < 3:
    print("\nUpsampling minority classes...")
    # Upsample each product to have at least 3 samples
    upsampled_data = []
    for product in high_demand_data['Product'].unique():
        product_data = high_demand_data[high_demand_data['Product'] == product]
        if len(product_data) < 3:
            # Resample with replacement to get 3 samples
            product_data_upsampled = resample(product_data, 
                                             replace=True, 
                                             n_samples=3, 
                                             random_state=42)
            upsampled_data.append(product_data_upsampled)
        else:
            upsampled_data.append(product_data)
    
    high_demand_data = pd.concat(upsampled_data)
    print("After upsampling:")
    print(high_demand_data['Product'].value_counts())

# Prepare features and targets for product prediction
product_features = ['Location_encoded', 'Weather_encoded', 'Category_encoded', 
                   'Temperature(°C)', 'Humidity(%)', 'Temp_Humidity_Interaction',
                   'Location_Weather_Interaction']
X_product = high_demand_data[product_features]
y_product = high_demand_data['Product_encoded']

# Split data for product prediction - use smaller test size due to limited data
test_size = max(0.2, 1/len(y_product.unique()))  # Ensure at least 1 sample per class in test
test_size = min(test_size, 0.3)  # Cap at 30%
print(f"\nUsing test size: {test_size:.2f}")

X_train_prod, X_test_prod, y_train_prod, y_test_prod = train_test_split(
    X_product, y_product, test_size=test_size, random_state=42, stratify=y_product
)

# Scale features for product model
product_scaler = StandardScaler()
X_train_prod_scaled = product_scaler.fit_transform(X_train_prod)
X_test_prod_scaled = product_scaler.transform(X_test_prod)

# Use a simpler model due to limited data
print("\nTraining product prediction model...")
# Simplified hyperparameter grid due to small dataset
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Use stratified k-fold for cross-validation
cv = StratifiedKFold(n_splits=min(3, len(y_train_prod.unique())), shuffle=True, random_state=42)

product_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid, cv=cv, n_jobs=-1, verbose=1)
product_grid.fit(X_train_prod_scaled, y_train_prod)
product_model = product_grid.best_estimator_

print("\nBest parameters for product model:", product_grid.best_params_)

# Evaluate product prediction with cross-validation
cv_scores = cross_val_score(product_model, X_train_prod_scaled, y_train_prod, cv=cv)
print(f"Cross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

y_pred_prod = product_model.predict(X_test_prod_scaled)
product_accuracy = accuracy_score(y_test_prod, y_pred_prod)
print(f"Product Prediction Accuracy: {product_accuracy:.2%}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_prod, y_pred_prod, 
                           target_names=le_product.inverse_transform(np.unique(y_test_prod))))

# Prepare features and targets for price prediction
price_features = ['Location_encoded', 'Weather_encoded', 'Product_encoded', 
                 'Category_encoded', 'Temperature(°C)', 'Humidity(%)']
X_price = high_demand_data[price_features]
y_price = high_demand_data[['Suggested Price Min (₹/kg)', 'Suggested Price Max (₹/kg)']]

# Split data for price prediction
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price, y_price, test_size=test_size, random_state=42
)

# Scale features for price prediction
price_scaler = StandardScaler()
X_train_price_scaled = price_scaler.fit_transform(X_train_price)
X_test_price_scaled = price_scaler.transform(X_test_price)

# Train Random Forest Regressor for price prediction with simplified hyperparameter tuning
print("\nTraining price prediction model...")
price_param_grid = {
    'estimator__n_estimators': [50, 100],
    'estimator__max_depth': [None, 10],
    'estimator__min_samples_split': [2, 5],
    'estimator__min_samples_leaf': [1, 2]
}

price_grid = GridSearchCV(MultiOutputRegressor(RandomForestRegressor(random_state=42)), 
                         price_param_grid, cv=3, n_jobs=-1, verbose=1)
price_grid.fit(X_train_price_scaled, y_train_price)
price_model = price_grid.best_estimator_

print("\nBest parameters for price model:", price_grid.best_params_)

# Evaluate price prediction
y_pred_price = price_model.predict(X_test_price_scaled)
min_mae = mean_absolute_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0])
max_mae = mean_absolute_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1])
min_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Min (₹/kg)'], y_pred_price[:, 0]))
max_rmse = np.sqrt(mean_squared_error(y_test_price['Suggested Price Max (₹/kg)'], y_pred_price[:, 1]))

print("\nPrice Prediction Evaluation:")
print(f"Min Price - MAE: {min_mae:.2f}, RMSE: {min_rmse:.2f}")
print(f"Max Price - MAE: {max_mae:.2f}, RMSE: {max_rmse:.2f}")

# Feature importance visualization
plt.figure(figsize=(12, 8))
feat_importances = pd.DataFrame(product_model.feature_importances_, index=product_features)
feat_importances.sort_values(by=0).plot(kind='barh')
plt.title('Feature Importance for Product Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Prediction function
def predict_product_and_price(location, weather):
    # Encode inputs
    try:
        loc_encoded = le_location.transform([location])[0]
        weather_encoded = le_weather.transform([weather])[0]
    except ValueError as e:
        return f"Error: {e}. Please check your inputs."
    
    # Get temperature and humidity for the location and weather (using mean values)
    location_weather_data = data[(data['Location'] == location) & (data['Weather'] == weather)]
    if len(location_weather_data) == 0:
        # If no exact match, use overall means
        temp = data['Temperature(°C)'].mean()
        humidity = data['Humidity(%)'].mean()
    else:
        temp = location_weather_data['Temperature(°C)'].mean()
        humidity = location_weather_data['Humidity(%)'].mean()
    
    # Create interaction features
    temp_humidity_interaction = temp * humidity
    location_weather_interaction = loc_encoded * weather_encoded
    
    # Get category for the location (using most common category)
    location_data = data[data['Location'] == location]
    if len(location_data) > 0:
        most_common_category = location_data['Category'].mode()[0]
        category_encoded = le_category.transform([most_common_category])[0]
    else:
        category_encoded = 0  # Default to first category
    
    # Create DataFrame with all features for product prediction
    product_input = pd.DataFrame([[loc_encoded, weather_encoded, category_encoded, 
                                   temp, humidity, temp_humidity_interaction, 
                                   location_weather_interaction]], 
                                 columns=product_features)
    
    # Scale features for product prediction
    product_input_scaled = product_scaler.transform(product_input)
    
    # Predict product
    product_encoded = product_model.predict(product_input_scaled)[0]
    product = le_product.inverse_transform([product_encoded])[0]
    
    # Create input for price prediction
    price_input = pd.DataFrame([[loc_encoded, weather_encoded, product_encoded, category_encoded, temp, humidity]], 
                              columns=price_features)
    
    # Scale features for price prediction
    price_input_scaled = price_scaler.transform(price_input)
    
    # Predict price
    price = price_model.predict(price_input_scaled)[0]
    min_price, max_price = price[0], price[1]
    
    return {
        'Location': location,
        'Weather': weather,
        'High Demand Product': product,
        'Category': le_category.inverse_transform([category_encoded])[0],
        'Suggested Price Range': f"₹{min_price:.1f} - ₹{max_price:.1f} per kg"
    }

# Example usage
print("\nPrediction:")
print(predict_product_and_price('Chennai', 'Mist'))
print(predict_product_and_price('Coimbatore', 'Cloudy, chance of rain'))
'''
