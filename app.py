import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

model        = pickle.load(open('models/model.pkl',        'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))
le           = pickle.load(open('models/label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── All numerical inputs ───────────────────────────────────────
        Dosage_mg                      = float(request.form['Dosage_mg'])
        Abuse_Potential_Score          = float(request.form['Abuse_Potential_Score'])
        Regulatory_Risk_Score          = float(request.form['Regulatory_Risk_Score'])
        Adverse_Event_Reports          = float(request.form['Adverse_Event_Reports'])
        Price_Per_Unit                 = float(request.form['Price_Per_Unit'])
        Annual_Sales_Volume            = float(request.form['Annual_Sales_Volume'])
        Recall_History_Count           = float(request.form['Recall_History_Count'])
        Doctor_Recommendation_Rate     = float(request.form['Doctor_Recommendation_Rate'])
        Production_Cost                = float(request.form['Production_Cost'])
        Side_Effect_Severity_Score     = float(request.form['Side_Effect_Severity_Score'])
        Brand_Reputation_Score         = float(request.form['Brand_Reputation_Score'])
        Marketing_Spend                = float(request.form['Marketing_Spend'])
        Insurance_Coverage_Percentage  = float(request.form['Insurance_Coverage_Percentage'])
        Export_Percentage              = float(request.form['Export_Percentage'])
        Prescription_Rate              = float(request.form['Prescription_Rate'])
        Online_Sales_Percentage        = float(request.form['Online_Sales_Percentage'])
        Pharmacy_Distribution_Percentage = float(request.form['Pharmacy_Distribution_Percentage'])
        Approval_Time_Months           = float(request.form['Approval_Time_Months'])
        Competitor_Count               = float(request.form['Competitor_Count'])
        Hospital_Distribution_Percentage = float(request.form['Hospital_Distribution_Percentage'])
        RD_Investment_Million          = float(request.form['RD_Investment_Million'])
        Patent_Duration_Years          = float(request.form['Patent_Duration_Years'])
        Clinical_Trial_Phase           = float(request.form['Clinical_Trial_Phase'])

        # ── All categorical inputs ─────────────────────────────────────
        Drug_Form                = request.form['Drug_Form']
        Therapeutic_Class        = request.form['Therapeutic_Class']
        Manufacturing_Region     = request.form['Manufacturing_Region']
        Requires_Cold_Storage    = request.form['Requires_Cold_Storage']
        OTC_Flag                 = request.form['OTC_Flag']
        High_Risk_Substance      = request.form['High_Risk_Substance']

        # ── Column order must match EXACTLY how you trained ────────────
        outlier_cols     = ['Annual_Sales_Volume', 'Recall_History_Count', 'Adverse_Event_Reports']
        normal_num_cols  = ['Dosage_mg', 'Abuse_Potential_Score', 'Regulatory_Risk_Score',
                            'Price_Per_Unit', 'Doctor_Recommendation_Rate', 'Production_Cost',
                            'Side_Effect_Severity_Score', 'Brand_Reputation_Score',
                            'Marketing_Spend', 'Insurance_Coverage_Percentage',
                            'Export_Percentage', 'Prescription_Rate', 'Online_Sales_Percentage',
                            'Pharmacy_Distribution_Percentage', 'Clinical_Trial_Phase',
                            'Approval_Time_Months', 'Competitor_Count',
                            'Hospital_Distribution_Percentage', 'R&D_Investment_Million',
                            'Patent_Duration_Years']
        cat_cols         = ['Drug_Form', 'Therapeutic_Class', 'Manufacturing_Region',
                            'Requires_Cold_Storage', 'OTC_Flag', 'High_Risk_Substance']

        input_data = pd.DataFrame([[
            Annual_Sales_Volume, Recall_History_Count, Adverse_Event_Reports,
            Dosage_mg, Abuse_Potential_Score, Regulatory_Risk_Score,
            Price_Per_Unit, Doctor_Recommendation_Rate, Production_Cost,
            Side_Effect_Severity_Score, Brand_Reputation_Score,
            Marketing_Spend, Insurance_Coverage_Percentage,
            Export_Percentage, Prescription_Rate, Online_Sales_Percentage,
            Pharmacy_Distribution_Percentage, Clinical_Trial_Phase,
            Approval_Time_Months, Competitor_Count,
            Hospital_Distribution_Percentage, RD_Investment_Million,
            Patent_Duration_Years,
            Drug_Form, Therapeutic_Class, Manufacturing_Region,
            Requires_Cold_Storage, OTC_Flag, High_Risk_Substance
        ]], columns=outlier_cols + normal_num_cols + cat_cols)

        # Fix R&D column name back
        input_data.rename(columns={'R&D_Investment_Million': 'R&D_Investment_Million'}, inplace=True)

        transformed      = preprocessor.transform(input_data)
        prediction_num   = model.predict(transformed)[0]
        prediction_label = le.inverse_transform([prediction_num])[0]
        proba            = model.predict_proba(transformed)[0]
        confidence       = round(max(proba) * 100, 2)

        return render_template('index.html',
                               prediction=prediction_label,
                               confidence=confidence,
                               error=None)
    except Exception as e:
        return render_template('index.html',
                               prediction=None,
                               confidence=None,
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
