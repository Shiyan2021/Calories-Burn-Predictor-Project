from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('calories_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def recommend_meal(calories):
    if calories <= 200:
        return "Try a fruit salad, yogurt, or a light protein bar!"
    elif calories <= 400:
        return "Consider a smoothie, egg sandwich, or peanut butter toast."
    elif calories <= 600:
        return "You might enjoy a grilled chicken wrap or quinoa salad."
    else:
        return "Refuel with a rice bowl, salmon with veggies, or a healthy burrito!"



@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = 1 if request.form['gender'] == 'Male' else 0
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(features)[0]

        meal = recommend_meal(prediction)

        return render_template('result.html',
                               prediction_text=f'Calories Burned: {prediction:.2f} cal',
                               meal_text=f'Meal Suggestion: {meal}')
    except Exception as e:
        return render_template('result.html', prediction_text=f'Error: {str(e)}')



if __name__ == '__main__':
    app.run(debug=True)
