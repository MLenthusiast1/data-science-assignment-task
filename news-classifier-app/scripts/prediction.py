import pandas as pd
import sys
import joblib
from sklearn.metrics import precision_score,recall_score,accuracy_score

print('reading data...')
BaseDir=sys.argv[1]
inputpath=BaseDir+"/DataInput/"
modellocation=BaseDir+"/TrainingOutput/"
outputpath=BaseDir+"/PredictionOutput/"

#Reading data to predict
df=pd.read_csv(inputpath+"test_data_redacted.csv",sep='\t')
X_test=df['text']
y_test=df['category']


#Load Model
print('loading model pipeline...')
model=joblib.load(open(modellocation+"Model10192021.sav","rb"))
y_pred=model.predict(X_test)


print('writing prediction output to csv...')
output=pd.DataFrame()
output['text']=X_test
output['category_predicted']=y_pred
output.to_csv(outputpath+"prediction_output.csv")
print('prediction output written to disk successfully!!!')
