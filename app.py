# importing the requirements
import numpy as np
import pickle
from flask import Flask, request, render_template

# initializing my app name with flask
# this acts like our WSGI(website gateway interface), a standard which is used to communicate between webserver and web application
app = Flask(__name__)

# this loads the model to our website
model = pickle.load(open("ufo-model.pkl", "rb"))

# decorater 
@app.route('/')#URL here which calls the index.html rendering the template
def home():
    return render_template("index.html")

# this decorator adds predict to the URL(which is another URL here)
@app.route('/predict', methods=["POST"])
def predict():
    # this gets the user values as a seperate list
    int_features = [int(x) for x in request.form.values()]
    
    # we are converting that list to array
    final_features = [np.array(int_features)]
    
    # the test data, which is final_features here is considered
    prediction = model.predict(final_features)
    
    # as the country is labelcoded to numbers, we get the exact number by getting through index
    output = prediction[0]

    # we set countries list according to our label encoded numbers
    countries = ["Australia","Cannada","Germany","UK","US"]

    # Finally, returning our predicted value!
    return render_template(
        "index.html", prediction_text = 'Likely country: {}'.format(countries[output])
    )

if __name__=='__main__':
    app.run(debug=True)
