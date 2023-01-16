from flask  import Flask,app,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import sklearn    
import pickle




app=Flask(__name__)
scaller=pickle.load(open("scalling.pkl","rb"))

regmodel=pickle.load(open("bost_pred_pkl","rb"))


@app.route('/')
def home():
    return render_template("home.html")
@app.route("/predict_api",methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values()).reshape(1,-1)))
    new_data=scaller.transform(np.array(list(data.values()).reshape(1,-1)))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_result=scaller.transform(np.array(data).reshape(1,-1))
    print(final_result)
    output=regmodel.predict(final_result)[0]
    return render_template("home.html",prediction_text="The House price pridication is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)

    




