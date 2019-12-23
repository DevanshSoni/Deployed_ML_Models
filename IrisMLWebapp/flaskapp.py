from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

app=Flask(__name__)

@app.route('/')
def Welcome():
    return render_template('index.html')

@app.route('/check',methods=['POST'])
def predictvalues():
    slen=float(request.form.get('slen'))
    swid=float(request.form.get('swid'))
    plen=float(request.form.get('plen'))
    pwid=float(request.form.get('pwid'))
    data=[np.array([slen,swid,plen,pwid])]

    with open('LogisticRegressionModel.pkl','rb') as fp:
        model=pkl.load(fp)
    
    predicted=model.predict(data)
    return render_template('index.html',predicted_text="Predicted Flower Value is {}".format(*predicted))

if __name__=='__main__':
    app.run()