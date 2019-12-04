# Sentiment Analysis scikit learn model

This is the sentiment analysis model that was trained in scikit learn. It is a Logistic Regression model trained on the twitter dataset

## How to use

```py
import pickle
from urllib.request import Request, urlopen
modelreq = Request("https://raw.githubusercontent.com/SharanSMenon/ml-models/master/sentiment_analysis/scikit-learn/sentiment_model.p") # Getting the model
vectorreq = Request("https://raw.githubusercontent.com/SharanSMenon/ml-models/master/sentiment_analysis/scikit-learn/vectorizer.p") # Getting the vectorizer

m = urlopen(modelreq)
v = urlopen(vectorreq)

model = pickle.load(m)
vectorizer = pickle.load(v)

def predict(s):
    sa = [s]
    transformed = vectorizer.transform(sa).toarray()
    pred = model.predict(transformed)
    return pred[0]
    
print(predict("This sucks")) # returns "neg"
print(predict("This is great")) # returns "pos"
```

The model either returns `neg` or `pos`.

The model has an accuracy of around 80%.
