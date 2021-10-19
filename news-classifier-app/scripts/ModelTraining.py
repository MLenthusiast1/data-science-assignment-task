import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score,accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from datetime import datetime
import joblib
import re
import string

print('Model training script started...')
now =datetime.now()
DateOfExec= now.strftime('%m%d%Y')

print('Data Reading...')
BaseDir=sys.argv[1]
inputpath=BaseDir+"/DataInput/"
outputpath=BaseDir+"/TrainingOutput/"

# Reading Data
df=pd.read_csv(inputpath+"data_redacted.tsv",sep='\t')
NoOfRows=df.shape[0]


print('Data cleaning..')
df.drop(columns=['url','title'],inplace=True)

def text_cleaner(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('\n','',text)
    return text

cleaner=lambda x:text_cleaner(x)
df['cleaned_text']=pd.DataFrame(df['text'].apply(cleaner))
df.head()

#spliting train and test data
df_train=df[0:int(NoOfRows*0.80)]
df_test=df[int(NoOfRows*0.80):NoOfRows]
df_test.to_csv(inputpath+"test_data_redacted.csv")

print('Data processing...')
X=df_train['cleaned_text']
y=df_train['category']

#Initializing TfidfVectorizer
vect=TfidfVectorizer(stop_words='english')
X_vec= vect.fit_transform(X)

print('Oversampling training data...')
#Initializing SOMTE oversampling technique
smote=SMOTE(random_state=42)
X_sampled, y_sampled= smote.fit_resample(X_vec, y)

print('Splitting data in train and test...')
#Splitting in train and test
X_train,X_test,y_train,y_test=train_test_split(X_sampled,y_sampled,random_state=44,test_size=0.25)

print('model training...')
#Initializing a Support Vector Classifier
svc=SVC()
svc.fit(X_train,y_train)
print('model is trained successfully!!!')
y_pred=svc.predict(X_test)
print('Evaluating performance matrics of model...')
print('accuracy_score : ',accuracy_score(y_test,y_pred))
print('precision_score : ',precision_score(y_test,y_pred,average='weighted'))
print('recall_score : ',recall_score(y_test,y_pred,average='weighted'))
print('writing model pipeline to disk...')
pipeline=Pipeline([('vectorizer',vect),('classifier',svc)])

# writing pipeling to disk
with(open(outputpath+"Model"+DateOfExec+".sav","w")) as fp:
    pass
joblib.dump(pipeline,open(outputpath+"Model"+DateOfExec+".sav","wb"))

print('model pipeline is written to disk successfully!!')