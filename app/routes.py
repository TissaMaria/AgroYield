from flask import Blueprint, render_template, request
import pandas as pd
from model.crop_model import predict_yield

main = Blueprint('main', __name__)
@main.route('/home')
def home():
    return render_template('home.html')


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting all form data and converting them to appropriate types
        data = {
            'District_Name': request.form['district_name'],
            
            'Season': request.form['season'],  # Season as string
            'Crop': request.form['crop_name'],  # Crop name as string
            'Area': float(request.form['area']),  # Area in hectares as float
            
            'N': float(request.form['n']),  # Nitrogen content as float
            'P': float(request.form['p']),  # Phosphorus content as float
            'K': float(request.form['k']),  # Potassium content as float
            'Temperature': float(request.form['temperature']),  # Temperature in °C as float
            'Humidity': float(request.form['humidity']),  # Humidity in percentage as float
            'Ph': float(request.form['ph']),  # pH value as float
            'Rainfall': float(request.form['rainfall'])  # Rainfall in mm as float
        }
        
        # Use the predict_yield function to predict the crop yield based on all inputs
        yield_prediction = predict_yield(data)
        
        # Render the result page with the predicted yield
        return render_template('result.html', prediction=yield_prediction)
