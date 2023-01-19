
from flask import request, jsonify
import pickle
import pandas as pd
import flask
import os 
import re
import nltk
nltk.download('stopwords')

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords



os.chdir(os.path.dirname(__file__))


application = flask.Flask(__name__)



conversacion={'Introduccion':'''Hola, soy SARA. ¿En qué puedo ayudarte? Por favor, introduce brevemente qué te preocupa y veré que puedo hacer. Todavía estoy en desarrollo, por lo que te agradecería que lo comentases en una única oración''',

             0: {'Respuesta': '''Vaya, parece que conoces a alguien que lo está pasando mal. ¿Es así?'''},

             1: {'Respuesta': '''Por lo que entiendo creo que te vendría bien un poco de ayuda externa. ¿Quieres ver qué recursos tienes a tu disposición?'''},

             2: {'Respuesta': '''Es una situación complicada, a veces difícil reconocer lo que tenemos delante. ¿Me dejarías hacerte unas preguntas para conocer mejor tu situación?'''}
            }


@application.route('/')
def main():
    return 'PÁGINA PRINCIPAL'


# PREDICCIÓN
"""
La petición sería:
http://127.0.0.1:5000/prediccion?question=Frase
"""

@application.route('/prediccion', methods=['GET'])
def predict():
    model = pickle.load(open(r'model_jhon.pkl', 'rb'))
    #Quita tildes
    def normalize(text):
        replacement = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u")),
        for i in replacement: 
            for a, b in i: 
                text = text.replace(a, b).replace(a.upper(), b.upper())
        return text

    #Quita signos de puntuación y mayúsculas
    signos = re.compile("(\.)|(\;)|(\:)|(\/)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    def signs_texts(text):
        return signos.sub(' ', text.lower())

    #Simplificar el texto todo lo posible. 
    #Stopwords
    from nltk.corpus import stopwords

    spanish_stopwords = stopwords.words('spanish')
    def remove_stopwords(df):
        #print([word for word in df.split() if word not in spanish_stopwords])
        return " ".join([word for word in df.split() if word not in spanish_stopwords])

    #snowball
    from nltk.stem.snowball import SnowballStemmer

    def spanish_stemmer(x):
        stemmer = SnowballStemmer('spanish')
        return " ".join([stemmer.stem(word) for word in x.split()])


    data = str(request.args["question"])
    data= signs_texts(data)
    data= normalize(data)
    data= remove_stopwords(data)
    data= spanish_stemmer(data)
    prediction = model.predict(pd.Series(data))
    prediction = str(prediction[0])
    if prediction == '0':
        return conversacion[0]['Respuesta']
    if prediction == '1':
        return conversacion[1]['Respuesta']
    else:
        return conversacion[2]['Respuesta']
    # return jsonify({'response' : str(prediction[0])})


if __name__ == "__main__":    
    application.debug = True
    application.run()
