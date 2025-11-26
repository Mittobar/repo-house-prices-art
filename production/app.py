from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import joblib


app=Flask(__name__)

modelo_hpp= (__name__)

modelo_hpp=joblib.load("../models/house_prices_pipeline.pkl")

@app.route("/predict_single", methods=['POST'])
def predict_singel():
    data=request.get_json()
    df_data=pd.DataFrame(data)
    prediccion=modelo_hpp.predict(df_data)
    print(prediccion)
    print(df_data)
    return"Hola"

#forma 1
@app.route("/saludar", methods =['GET'])
def saludo_v1():
    return "Hola a todos"

@app.route("/saludar_v2/<nombre>", methods =['GET'])
def saludo_v2(nombre):
    return f"Hola a {nombre}"

#forma 2
@app.route ("/suma", methods=['GET']) #datos
def sumar():
    x=request.args.get("x", type=int)
    y=request.args.get("y", type=int)
    return f"La suma de los valores es :{x +y}"

#forma 3 más seguridad
@app.route("/restar", methods=['POST']) #proceso
def resta_numeros():
    data=request.get_json()

    x=data.get("x")
    y=data.get("y")
    output= jsonify({"x":x, "y":y,"resultado": x-y})
    return output



if __name__ == "__main__":
    app.run(debug=True)
