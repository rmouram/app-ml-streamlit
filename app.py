import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#título
st.write("Prevendo Diabetes")

#dataset
df = pd.read_csv("data/diabetes.csv")

#cabeçalho
st.subheader("Informações dos dados")

#nomedousuário
user_input = st.sidebar.text_input("Digite seu nome")

#escrevendo o nome do usuário
st.write("Paciente:", user_input)

#dados de entrada
x = df.drop(['Outcome'],1)
y = df['Outcome']

#separa dados em treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#dados dos usuários com a função
def get_user_date():
    pregnancies = st.sidebar.slider("Gravidez",0, 15, 1)
    glicose = st.sidebar.slider("Glicose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Pressão Sanguínea", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Espessura da pele", 0, 99, 20)
    insulin = st.sidebar.slider("Insulina", 0, 900, 30)
    bni= st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)
    age = st.sidebar.slider ("Idade", 15, 100, 21)

    #dicionário para receber informações
    user_data = {
    'Gravidez': pregnancies,
    'Glicose': glicose,
    'Pressão Sanguínea': blood_pressure,
    'Espessura da pele': skin_thickness,
    'Insulina': insulin,
    'Índice de massa corporal': bni,
    'Histórico familiar de diabetes': dpf,
    'Idade': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_input_variables = get_user_date()

#grafico
graf = st.bar_chart(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(x_train, y_train)

#acurácia do modelo
st.subheader('Acurácia do modelo')
st.write(accuracy_score(y_test, dtc.predict(x_test))*100)

#previsão do resultado
prediction = dtc.predict(user_input_variables)

st.subheader('Previsão:')
st.write(prediction)

