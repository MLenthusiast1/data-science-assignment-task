import pickle
from flask import Flask, request,render_template


app = Flask(__name__)
model=pickle.load(open('classifer_model.pkl','rb'))


@app.route('/')
def home():
      return render_template('index.html')


@app.route("/category",methods=['POST'])
def category():
      news_text = request.form.get('text')
      print('news_text : ',news_text)
      predict_feature=[news_text]
      prediction = model.predict(predict_feature)
      output=prediction[0]
      print('Classifier output :',output)
      return render_template('index.html', prediction_text='News Category is : {}'.format(output))


@app.route("/category/api",methods=['POST'])
def category_api():
      news_text = request.form.get('text')
      print('news_text : ',news_text)
      predict_feature=[news_text]
      prediction = model.predict(predict_feature)
      output = prediction[0]
      print('Classifier output :', output)
      return output
if __name__ == '__main__':
    app.run(debug=True)
