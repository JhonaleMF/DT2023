
from flask import request, jsonify
import pickle
import pandas as pd
import flask
import os 
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')



os.chdir(os.path.dirname(__file__))




# Con esta funcion dejamos solo las raices de las palabrasssssssssssssssss
def stem_text(df):
    stemmer = SnowballStemmer(language='spanish')
    df = df.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    return df
# Con esta funcion quitamos los signos de puntuacion
def depuntuation (df):
    df = df.apply(lambda x: x.replace('[{}]'.format(string.punctuation), ''))
    return df

# Con esta funcion quitamos las tildes
def remove_tildes(df):
    df = df.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    return df

# Con esta funcion quitamos los stopwords
def remove_stopwords(df):    
    spanish_stopwords = stopwords.words('spanish')
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (spanish_stopwords)]))
    return df

# Con esta funcion pasamos las celdas a minuscula
def to_lowercase(df):
    df = df.apply(lambda x: x.lower())
    return df

model = pickle.load(open(r'model.pkl', 'rb'))
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
    #Signos de puntuacion, tildes y minusculas
   

    # def depuntuation (df):
    #     df = df.apply(lambda x: x.replace('[{}]'.format(string.punctuation), ''))
    #     return df

    # def stem_text(df):
    #     stemmer = SnowballStemmer(language='spanish')
    #     df = df.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    #     return df

    # def remove_tildes(df):
    #     df = df.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    #     return df
    # def remove_stopwords(df):    
    #     spanish_stopwords = stopwords.words('spanish')
    #     df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (spanish_stopwords)]))
    #     return df
    # def to_lowercase(df):
    #     df = df.apply(lambda x: x.lower())
    #     return df


    data = str(request.args["question"])
    # data= depuntuation(data)
    # data= to_lowercase(data)
    # data= remove_tildes(data)
    # data= remove_stopwords(data)
    # data= stem_text(data)
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
