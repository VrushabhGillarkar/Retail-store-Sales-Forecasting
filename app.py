from flask import Flask, render_template, request
import pickle
import numpy as np

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    category = int(request.form['category'])
    price = float(request.form['price'])
    month = int(request.form['month'])
    day_of_week = int(request.form['day_of_week'])
    day = int(request.form['day'])
    year = int(request.form['year'])

    input_features = np.array([[gender, age, category, price, month, day_of_week, day, year]])
    prediction = model.predict(input_features)[0]

    return render_template('result.html', 
                           prediction=prediction,
                           gender=gender,
                           age=age,
                           category=category,
                           price=price,
                           month=month,
                           day_of_week=day_of_week,
                           day=day,
                           year=year)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
