# Language Detection

This a a simple language detection model. It can classify text in the following languages:

1. English
2. Spanish
3. German
4. Portugese
5. Arabic
6. Russian
7. Dutch
8. French
9. Italian
10. Polish
11. Japanese

The model was trained with sklearn and it has a 95% accuracy.

## Script for usage

> Make sure that you have scikit-learn installed

```py
import pickle
from urllib.request import Request, urlopen
modelreq = Request("https://raw.githubusercontent.com/SharanSMenon/ml-models/master/language_detection/sklearn/language_detection.p")
m = urlopen(modelreq)
langs_dict = { 'ar': "Arabic", 'es': 'Spanish', 'en': 'English',
    'fr': 'French', 'de': 'German', 'it': 'Italian',
    'ja': 'Japanese', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portugese', 'ru': 'Russian'
}
langs = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ja', 'nl', 'pl', 'pt', 'ru']
model = pickle.load(m)
def predict(sent):
    predicted = model.predict([sent])
    lang = langs[predicted[0]]
    lg = langs_dict[lang]
    return lg
print(predict("This is english")) # English
print(predict("Hola, como estas")) # Spanish
print(predict("Dies ist ein Test, um die Sprache zu erkennen.")) # German
print(predict("Ceci est un test de d\xe9tection de la langue.")) # French
```
