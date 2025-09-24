🌾 ClimaMart AI – Product Demand & Price Prediction

This project uses Machine Learning to predict high-demand agricultural products and their dynamic price ranges based on location and weather conditions. It also suggests medium-demand products available under the same conditions.

The system is powered by Random Forest Classifier for product prediction and Random Forest Regressor (MultiOutputRegressor) for price prediction.


📌 Features

* ✅ **Product Prediction** – Predicts which product will have **high demand** given a location and weather condition.
* ✅ **Price Range Prediction** – Suggests **minimum and maximum prices (₹/kg)** for high-demand products.
* ✅ **Medium-Demand Suggestions** – Lists medium-demand products with average price ranges for the same conditions.
* ✅ **Interactive Input** – Users can select location and weather condition from available options.
* ✅ **Model Evaluation** – Accuracy, MAE, and RMSE metrics included.


📂 Project Structure

├── data.csv                 # Dataset with products, weather, location, demand levels, and price ranges
├── main.py                  # Core script for training models and making predictions
├── README.md                # Documentation (this file)
└── feature_importance.png   # Visualization of feature importance (optional, advanced version)


🛠️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/climamart-ai.git
   cd climamart-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset (`data.csv`) in the project root directory.

4. Run the program:

   ```bash
   python main.py
   ```


📊 Dataset Format

The dataset (`data.csv`) must contain the following columns:

* `Location` – City or region (e.g., Chennai, Coimbatore)
* `Weather` – Weather condition (e.g., Mist, Cloudy, Sunny)
* `Category` – Product category (e.g., Vegetables, Fruits)
* `Product` – Product name (e.g., Tomato, Mango)
* `Demand Level` – High / Medium / Low
* `Suggested Price Min (₹/kg)` – Minimum suggested price per kg
* `Suggested Price Max (₹/kg)` – Maximum suggested price per kg
* `Temperature(°C)` – Temperature in Celsius
* `Humidity(%)` – Humidity percentage


🚀 How It Works

1. Data Preprocessing

   * Label encoding for categorical variables (Location, Weather, Product, Category, Demand).
   * Interaction features (Temperature × Humidity, Location × Weather).

2. Model Training

   * Random Forest Classifier predicts high-demand products.
   * Random Forest Regressor (MultiOutput) predicts min & max prices.

3. Prediction

   * User provides Location and Weather.
   * Model outputs:

     * 🎯 High Demand Product
     * 💰 Suggested Price Range
     * 📦 Medium Demand Products (if available)


📈 Example Output
Product Prediction Accuracy: 66.67%

Price Prediction Evaluation:
Min Price - MAE: 1.82, RMSE: 1.89
Max Price - MAE: 2.38, RMSE: 2.56

Available locations:
1. Ariyalur
2. Chengalpattu
3. Chennai
4. Coimbatore
5. Cuddalore
6. Dharmapuri
7. Dindigul
8. Erode
9. Kancheepuram
10. Kanyakumari
11. Karur
12. Krishnagiri
13. Madurai
14. Mayiladuthurai
15. Nagapattinam
16. Nilgiris
17. Perambalur
18. Pudukkottai
19. Ramanathapuram
20. Ranipet
21. Salem
22. Sivaganga
23. Tenkasi
24. Thanjavur
25. Theni
26. Thoothukudi
27. Tiruchirappalli
28. Tirunelveli
29. Tirupathur
30. Tiruppur
31. Tiruvallur
32. Tiruvannamalai
33. Tiruvarur
34. Vellore
35. Viluppuram
36. Virudhunagar

Enter location (name or number): 4

Available weathers:
1. Cloudy, chance of rain
2. Cloudy, isolated rain
3. Mist
4. Partly Cloudy

Enter weather (name or number): 1

==================================================
PREDICTION RESULTS FOR COIMBATORE
==================================================
PREDICTION RESULTS FOR COIMBATORE
Weather Condition: Cloudy, chance of rain

HIGH DEMAND PRODUCT:
  Product: Chilli
  Suggested Price Range: ₹81.5 - ₹101.4 per kg

MEDIUM DEMAND PRODUCTS:
  Product: Onion
  Suggested Price Range: ₹39.0 - ₹42.0 per kg
  Product: Papaya
  Suggested Price Range: ₹41.0 - ₹45.0 per kg

==================================================


📌 Future Enhancements

* 📡 Integrate real-time weather API for live predictions.
* 📊 Deploy as a Flask/Streamlit web app for user-friendly access.
* 🔮 Use Deep Learning models (LSTM, GRU) for time-series price forecasting.

* 🏪 Expand dataset with more locations, seasons, and crops.
