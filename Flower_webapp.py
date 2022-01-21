import streamlit as st
st.title("Flower Classification WebApp")
activities=['SVM','KNN','NB','DT','RF']
option=st.sidebar.selectbox('Which model you use?',activities)
sl=st.slider('Select sepal length', 0.0, 10.0)
sw=st.slider('Select sepal width', 0.0, 5.0)
pl=st.slider('Select petal length', 0.0, 10.0)
pw=st.slider('Select petal width', 0.0, 5.0)  
feature_list=[sl,sw,pl,pw]    
import numpy as np
single_pred = np.array(feature_list).reshape(1,-1)
clas=['setosa','versicolor','virginica']
import pickle
if st.button('Predict'):        
    if option=='SVM':
        SVM_model=pickle.load(open('SVM.pkl', 'rb'))
        st.success(clas[int(SVM_model.predict(single_pred))])
    elif option=='KNN':
        KNN_model=pickle.load(open('KNN.pkl', 'rb'))
        st.success(clas[int(KNN_model.predict(single_pred))])
    elif option=='NB':
        NB_model=pickle.load(open('NB.pkl', 'rb'))
        st.success(clas[int(NB_model.predict(single_pred))])
    elif option=='DT':
        DT_model=pickle.load(open('DT.pkl', 'rb'))
        st.success(clas[int(DT_model.predict(single_pred))])
    else:
        RF_model=pickle.load(open('RF.pkl', 'rb'))
        st.success(clas[int(RF_model.predict(single_pred))])

