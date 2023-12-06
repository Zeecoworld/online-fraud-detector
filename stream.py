import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

  # loading in the model to predict on the data 
pickle_in = open('online-fraud-model.joblib','rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
      return 'Welcome all'

  # defining the function which will make the prediction using 
  # the data which the user inputs 
def prediction(step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFlaggedFraud):  
    prediction = classifier.predict( 
          [[step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFlaggedFraud]]) 
    return prediction 

  # this is the main function in which we define our webpage 
def main(): 
          # giving the webpage a title 
    st.title("Online fraud Predictor") 
         
      # here we define some of the front end elements of the web page like 
      # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
      <div style ="background-color:yellow;padding:13px"> 
      <h1 style ="color:black;text-align:center;">Streamlit Online Fraud ML App </h1> 
      </div> 
      """
         
      # this line allows us to display the front end aspects we have 
      # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
         
      # the following lines create text boxes in which the user can enter 
      # the data required to make the prediction 
    step = st.text_input("step", "") 
    type = st.text_input("type", "") 
    amount = st.text_input("amount", "") 
    nameOrig = st.text_input("nameOrig", "") 
    oldbalanceOrg = st.text_input("oldbalanceOrg", "") 
    newbalanceOrig = st.text_input("newbalanceOrig", "") 
    nameDest = st.text_input("nameDest", "") 
    oldbalanceDest = st.text_input("oldbalanceDest", "") 
    newbalanceDest = st.text_input("newbalanceDest", "") 
    isFlaggedFraud = st.text_input("isFlaggedFraud", "") 
    result ="" 
         
      # the below line ensures that when the button called 'Predict' is clicked, 
      # the prediction function defined above is called to make the prediction 
      # and store it in the variable result 
    if st.button("Predict"): 
        result = prediction(step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFlaggedFraud) 
    st.success('The output is {}'.format(result)) 
       
if __name__=='__main__': 
    main()
