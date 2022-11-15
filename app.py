from flask import Flask, render_template, url_for, request
import pickle
import joblib

filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.json
        
        data = [message['message']]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    if my_prediction == 1:
        return {"message": "message contein insupported word .",
                "data":1}
    else:
        return {"message": "message clear.",
                "data":0}


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
