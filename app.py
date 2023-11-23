# Import relevant Packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle



# Preprocess the datas
# You may need to clean and tokenize the "Article" column here.

def preprocess_text(text):
  text = text.lower()
  text = text.replace(',', '')
  text = text.replace('.', '')
  # text = text.split()
  return text

def model_build( df, vectorizer ) : 
    df = df.copy()
    
    # Text Cleaning
    # print( len( user_article ) )
    df['Cleaned Heading'] = df['Heading'].apply( lambda x :  preprocess_text( str(x)  ) )
    
    x = vectorizer.fit_transform( df['Cleaned Heading'] )
    y = df['NewsType'].replace({"business"  :  0 , "sports" : 1})

    x_train, x_test, y_train , y_test = train_test_split(x, y, random_state=111, test_size=0.2)
    model = SVC()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print( "\nAccuracy_score : \n", round( accuracy_score(y_predict, y_test) , 2 ) )
    print( "\nclassification_report : \n", classification_report(y_predict, y_test) )

    return model


## Approach 1 : Only Backend (No Streamlit)
#####################################################################################
### Reading the dataset 

# Load your Excel data into a DataFrame
# dataset = pd.read_excel('Articles.xlsx')
# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()

# model = model_build( dataset , vectorizer )

#####################################################################################
## Working on Predictions 

# User's input article
# user_heading = """ percent lower  """

# user_heading_vector = vectorizer.transform( [user_heading] )
# news_type = model.predict(user_heading_vector)

# if news_type == 1 : print("News Type ==> sports")
# else : print("News Type ==> business")



## Approach 2 : Backend + frontend (Streamlit)
#####################################################################################
### Reading the dataset and 
# Making and Dumping the Model

# Load your Excel data into a DataFrame
# dataset = pd.read_excel('Articles.xlsx')
# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()

# model = model_build( dataset , vectorizer )

# pickle.dump(model, open("MyModel.pkl", "wb"))
# pickle.dump(vectorizer, open("Myvectorizer.pkl", "wb"))


#####################################################################################
## Working on Predictions 
# import time
# time.sleep(2)


# TF-IDF Vectorization
def getPredictions(text, model_Taken) : 
    # print("text : " , text)
    user_heading_vector = vectorizer_Taken.transform( [text] )
    # print("user_heading_vector : " , user_heading_vector)
    news_type = model_Taken.predict(user_heading_vector)
    # print("news_type : " , news_type)
    return news_type
   

if __name__ == "__main__" : 

  st.title("Machine Learning Project")

  # Get the user's input
  user_heading = st.text_input('Enter the news headline:', "")


  # User's input article
  # user_heading = """ percent lower  """

  model_Taken = pickle.load(open("MyModel.pkl", "rb"))
  vectorizer_Taken = pickle.load(open("Myvectorizer.pkl", "rb"))

  news_type = getPredictions(user_heading, model_Taken)

  displayOutput = st.empty()
  if news_type == 1 : displayOutput.write("News Type ==> sports")
  else : displayOutput.write("News Type ==> business")




