# property-price-predictor
 Property (Bengaluru) Price Predictor Machine learning model that estimates real estate prices using location, area type, size, and room features. Features: 
- 85%+ accurate Random Forest model 
- Streamlit web interface 
- Market trend visualizations 
- End-to-end pipeline from data to deployment  
 (Python - Scikit-learn - Streamlit)

## Dataset Provenance
This dataset was collected around 2017–2018, and therefore price estimates will naturally be under current market values. The purpose of this project is to demonstrate end-to-end preprocessing, feature engineering, regression modeling, and evaluation—not to predict current market rates.


## Installation
```bash
git clone https://github.com/<mohit-rahangdale>/property-price-predictor.git
cd property-price-predictor
pip install -r requirements.txt
python main.py
streamlit run app.py
