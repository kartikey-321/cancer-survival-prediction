from flask import Flask,render_template,request,redirect

import joblib
model=joblib.load('cancer_survival.pkl')

app=Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def submit():
    if request.method=='POST':
        age=int(request.form['age'])
        timerecurrence=float(request.form['timerecurrence'])
        diam=int(request.form['diam'])
        posnodes=int(request.form['posnodes'])
        grade=int(request.form['grade'])
        angioinv=int(request.form['angioinv'])
        lymphinfil=int(request.form['lymphinfil'])
        esr1=float(request.form['esr1'])
        histtype=int(request.form['histtype'])
        chemo=request.form['chemo']
        if chemo=='Yes':
            chemo=1
        else:
            chemo=0
        
        hormonal=request.form['hormonal']
        if hormonal=='Yes':
            hormonal=1
        else:
            hormonal=0

        amputation=request.form['amputation']
        if amputation=='Yes':
            amputation=1
        else:
            amputation=0

        result=model.predict([[age,timerecurrence,chemo,hormonal,amputation,histtype,diam,posnodes,grade,angioinv,lymphinfil,esr1]])
    
    return render_template('predict.html',result=result)





if __name__=='__main__':
    app.run(debug=True)