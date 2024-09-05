from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from wtforms import Form, FloatField, validators

# Load mo hinh da huan luyen
model = joblib.load('model/iris_classification_naive_bayes_model.pkl')

# Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')



# Form
class IrisForm(Form):
    sepal_length = FloatField('Sepal Length (cm)', [validators.InputRequired()])
    sepal_width = FloatField('Sepal Width (cm)', [validators.InputRequired()])
    petal_length = FloatField('Petal Length (cm)', [validators.InputRequired()])
    petal_width = FloatField('Petal Width (cm)', [validators.InputRequired()])


# Trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Nhan du lieu thong qua request.form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Dua du lieu vao mo hinh da huan luyen
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        # Chuyen doi thanh nhan
        species = ['Setosa', 'Versicolor', 'Virginica']
        prediction = species[prediction]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
