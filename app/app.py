# from flask import Flask,request,render_template
# import numpy as np
# import pandas as pd
# import sklearn
# import pickle

# # importing model
# model = pickle.load(open('C:\Desktop\Flask\model.pkl','rb'))
# sc = pickle.load(open('C:\Desktop\Flask\standscaler.pkl','rb'))
# ms = pickle.load(open('C:\Desktop\Flask\minmaxscaler.pkl','rb'))

# # creating flask app
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route("/predict",methods=['POST'])
# def predict():
#     N = request.form['Nitrogen']
#     P = request.form['Phosphorus']
#     K = request.form['Potassium']
#     temp = request.form['Temperature']
#     humidity = request.form['Humidity']
#     ph = request.form['Ph']
#     rainfall = request.form['Rainfall']

#     feature_list = [N, P, K, temp, humidity, ph, rainfall]
#     single_pred = np.array(feature_list).reshape(1, -1)

#     scaled_features = ms.transform(single_pred)
#     final_features = sc.transform(scaled_features)
#     prediction = model.predict(final_features)

#     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                  8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                  14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                  19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there".format(crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
#     return render_template('index.html',result = result)

# # python main
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

# Importing model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extracting form data
        features = [
            float(request.form['Nitrogen']),
            float(request.form['Phosphorus']),
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['Ph']),
            float(request.form['Rainfall'])
        ]

        # Transforming features
        single_pred = np.array(features).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        # Mapping prediction to crop
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        crop = crop_dict.get(prediction[0], "Unknown Crop")
        result = f"The best crop to cultivate is {crop}."
        
    except Exception as e:
        result = f"Error: {str(e)}"
        
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
