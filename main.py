from flask import Flask, jsonify, request
import os
from catboost import CatBoostRegressor

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    data = request.get_json()
    regTest = CatBoostRegressor()
    regTest.load_model('modeltest.cbm')
    result = regTest.predict([[data['Age'], data['BusinessTravel'], data['DailyRate'], data['Department'], data['Education'], data['EducationField'], data['Gender'], data['HourlyRate'], data['JobInvolvement'], data['JobLevel'], data['MaritalStatus'], data['MonthlyIncome'], data['NumCompaniesWorked'], data['OverTime'], data['StandardHours'], data['TotalWorkingYears'], data['YearsAtCompany'], data['YearsInCurrentRole'], data['YearsSinceLastPromotion'], data['YearsWithCurrManager']]])

    return str(abs(10 - result[0]))


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
