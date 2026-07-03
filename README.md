# Airline Passenger Satisfaction - Agile Data Science PMA

MRTB 2173, MSc Business Intelligence and Analytics, UTM.

Predicts airline passenger satisfaction so a CX team can prioritise
service improvements. Built across four sprints.

## Dashboard
(add Streamlit Cloud URL here after deploying)

## Structure
- notebooks/ADS_PMA1.ipynb — EDA, quality checks, models
- src/data_utils.py — load + clean pipeline
- src/validation.py — data validation script
- tests/test_validation.py — pytest suite
- .github/workflows/ci.yml — CI pipeline
- dashboard/app.py — Streamlit dashboard
- models/model.pkl, models/scaler.pkl — trained model + scaler

## Run locally
pip install -r requirements.txt
pytest tests/ -v
python src/validation.py
streamlit run dashboard/app.py

Dataset: Airline Passenger Satisfaction (Kaggle), 129,880 records, 24 columns.
