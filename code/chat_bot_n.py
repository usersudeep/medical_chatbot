import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import warnings
from flask import Flask, jsonify,render_template, request

warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('D:\health_prediction\code\Data\Training.csv')
testing= pd.read_csv('D:\health_prediction\code\Data\Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y
# print(training)
#added by manju
disease_input = ""
num_days = 0
symptoms_present = []
rslt = []

reduced_data = training.groupby(training['prognosis']).max()
# print(reduced_data)
#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
#training DecisionTreeClassifier module
clf = clf1.fit(x_train,y_train)

# print(clf.score(x_train,y_train))
# print ("cross result========")
#testing DecisionTreeClassifier module
scores = cross_val_score(clf, x_test, y_test, cv=3)
print ("scores.mean:")
print (scores.mean())

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

#functions starts
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    
    if((sum*days)/(len(exp)+1)>13):
        rslt.append("You should take the consultation from doctor. ")
#        print("You should take the consultation from doctor. ")
    else:
        rslt.append("It might not be that bad but you should take precautions.")
#        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('D:\health_prediction\code\MasterData\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('D:\health_prediction\code\MasterData\Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('D:\health_prediction\code\MasterData\symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list,inp):#converts input into reg pattern and check if it present or not
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
def sec_predict(symptoms_exp):
    df = pd.read_csv('D:\health_prediction\code\Data\Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      print(item)
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def TrainMdl():
    getSeverityDict()
    getDescription()
    getprecautionDict()

feature_name = []
def tree_to_code1(tree,feature_names,disease_input):
    tree_ = tree.tree_
    global feature_name 
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    chk_dis=",".join(feature_names).split(",") 

    conf,cnf_dis=check_pattern(chk_dis,disease_input)
    if conf==1:
        for num,it in enumerate(cnf_dis):
            print(num,")",it)
        return cnf_dis
    else:
        print("Enter valid symptom.")
        return 0

#user entered first symptoms used list of probal desises will be returned
def listdesises(symptom, name):
    global rslt
    rslt.clear()
    rslt.append("Mr./Mss. "+ name + " your Health Report");
    return tree_to_code1(clf,cols,symptom)

symptoms_given = []
present_disease = []
tree_ = clf.tree_   

def recurse(node, depth):
    global symptoms_present
    global disease_input
    global symptoms_given
    global present_disease
    global symptoms_exp

    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == disease_input:
            val = 1
        else:
            val = 0
        if  val <= threshold:
            recurse(tree_.children_left[node], depth + 1)
        else:
            symptoms_present.append(name)
            recurse(tree_.children_right[node], depth + 1)
    else:
        present_disease = print_disease(tree_.value[node])
        red_cols = reduced_data.columns 
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
        symptoms_exp=[]
        for syms in list(symptoms_given):
            symptoms_exp.append(syms)

def otherSymptoms(decese,nd):
    global disease_input
    global num_days
    global present_disease
    present_disease = []
    num_days = nd
    disease_input = decese
    recurse(0, 1)
    print("present_disease")
    print(present_disease)
    return symptoms_exp

def computeresult(symptoms):
    global num_days
   
    print(symptoms)
    second_prediction=sec_predict(symptoms)
    calc_condition(symptoms,num_days)
    if(present_disease[0]==second_prediction[0]):
        rslt.append("You may have "+ present_disease[0])
#        print("You may have ", present_disease[0])
        rslt.append(description_list[present_disease[0]])
#        print(description_list[present_disease[0]])
    else:
        rslt.append("You may have "+  present_disease[0]+ "or "+ second_prediction[0])
#        print("You may have ", present_disease[0], "or ", second_prediction[0])
        rslt.append(description_list[present_disease[0]])
#        print(description_list[present_disease[0]])
        rslt.append(description_list[second_prediction[0]])
#        print(description_list[second_prediction[0]])

#    this seems doble entry
#    rslt.append(description_list[present_disease[0]])
#    print(description_list[present_disease[0]])
    precution_list=precautionDictionary[present_disease[0]]
    rslt.append("Take following measures : ")
#   print("Take following measures : ")
    for  i,j in enumerate(precution_list):
#        print("(",i+1,")",j)
        rslt.append("(*)  "+ j )

    return rslt

TrainMdl()
print("----------------------------------------------------------------------------------------")
