<h1 align="center">IMAGE CLASSIFICATION IN HEALTHCARE</h1>

<p align="center">A Convolutional Neural Network (CNN) based project for classifying chest X-Rays as Covid-19 positive or negative.</p>

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)

## Introduction
This project aims to develop a model using Convolutional Neural Networks (CNN) to detect Covid-19 from chest X-Rays. The model is trained on a labeled dataset of chest X-Rays that are categorized as positive for Covid-19 or negative for Covid-19. The trained model can be used to classify new chest X-Rays and assist medical professionals in identifying potential Covid-19 cases.

## Getting Started
To set up this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/PRASOONR10/IMAGE-CLASSICATION-IN-HEATHCARE-COVID-19-`
2. Navigate to the project directory: `cd C:\Users\example\Documents\folder\IMAGE-CLASSICATION-IN-HEATHCARE-COVID-19-`
3. Make sure you have installed the required dependencies `pip install -r requirements.txt`

## Dataset
The dataset used for training the model can be found [here](https://www.kaggle.com/datasets/mr3suvhro/covid-19-xray-image-dataset-with-huge-samples). It consists of labelled chest X-Rays in two categories: Covid-19 positive and Covid-19 negative.

## Model Training
The CNN model is trained on the dataset using Google Colab. The code for model training can be found in the `train_model.ipynb` notebook. The trained model is then saved as `my_model.h5`.


## Usage
1. Open the Anaconda Powershell Prompt & navigate to the project directory, for example `cd C:\Users\example\Documents\folder\Covid-19-Detection`
2. Start the Flask web application: `python app.py`
3. Access the web application at `http://localhost:5000` in your web browser.
4. Upload a chest X-Ray image to the web application and click on the "Predict!" button.
5. The model will classify the X-Ray as Covid-19 positive or negative and display the result on the web page.
6. Some sample inputs are provided in the <i>sample-inputs</i> folder. 
