import json
from flask import Flask, redirect, render_template, request, url_for
from chat_bot_n import computeresult, listdesises, otherSymptoms

app = Flask(__name__,template_folder="static") 

@app.route('/')
def hello_world():
	return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['fname']
        semptom = request.form['smptm']
        deseses = listdesises(semptom,user)
        if deseses == 0:
             return render_template('index.html')
        else:
            return render_template('desese.html' , deseses = deseses)
    else:
        user = request.form['fname']
        semptom = request.form['smptm']
        return  render_template('desese.html' )
    
@app.route('/symptom' , methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        user_answer=request.form['op1']   
        ndays1 = request.form['ds1']
        ndays = int (ndays1)
        ans1 = user_answer
        ans2 = ans1.removesuffix('"')
        ans = ans2.removeprefix(' "')
        print("sympt:")
        print(ans)
        print("No:Days:")
        print(ndays)
        otrsympt = otherSymptoms(ans,ndays)
        print(otrsympt)
        return render_template('symptom.html',otrsympt = otrsympt)
    else:
        return  render_template('symptom.html' )

@app.route('/result' , methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        symptoms_exp1 = request.form.getlist("cb1")
        itms = []
        for itm in symptoms_exp1:
            tmp =  itm.replace('"','')
            tmp1= tmp.removeprefix(" ").removesuffix(" ")
            itms.append(tmp1)
        rslt = computeresult(itms)
        return render_template('result.html',rslt = rslt )
    else:
        return  render_template('result.html' )

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(debug=True)
