from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    name = request.form['name']
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.form['input_mail']
    input_data_features = vectorizer.transform([input_mail])
    prediction = model.predict(input_data_features)[0]
    
    if prediction == 1:
        return render_template('result_ham.html')
    else:
        return render_template('result_spam.html')

if __name__ == "__main__":
    # For running project locally comment the below line
    # app.run(debug=True)
    # For prod comment 
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

