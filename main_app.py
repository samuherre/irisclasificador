import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Clasificador IRIS", layout="wide")

st.title("🌸 Clasificador del Dataset IRIS")
st.write("Aplicación interactiva para aprender Machine Learning de forma pedagógica.")

# -----------------------------
# LOAD DATA
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = y

# -----------------------------
# SIDEBAR - CONFIGURACIÓN
# -----------------------------
st.sidebar.header("⚙️ Configuración")

model_option = st.sidebar.selectbox(
    "Selecciona un modelo",
    ["KNN", "SVM", "Árbol de Decisión"]
)

test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.2)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -----------------------------
# MODEL SELECTION
# -----------------------------
if model_option == "KNN":
    k = st.sidebar.slider("Número de vecinos (K)", 1, 10, 3)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "SVM":
    c = st.sidebar.slider("Parámetro C", 0.1, 10.0, 1.0)
    model = SVC(C=c)

elif model_option == "Árbol de Decisión":
    depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=depth)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# RESULTADOS
# -----------------------------
st.subheader("📊 Desempeño del Modelo")
st.write(f"**Accuracy:** {accuracy:.2f}")

# MATRIZ DE CONFUSIÓN
st.subheader("🔍 Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# -----------------------------
# VISUALIZACIÓN
# -----------------------------
st.subheader("📈 Visualización de Datos")

fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=y,
    palette="deep"
)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
st.pyplot(fig2)

# -----------------------------
# PREDICCIÓN MANUAL
# -----------------------------
st.subheader("🌱 Prueba tu propio dato")

input_data = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

if st.button("Predecir"):
    prediction = model.predict([input_data])
    predicted_class = target_names[prediction[0]]

    st.success(f"🌸 La especie predicha es: **{predicted_class}**")

# -----------------------------
# EXPLICACIÓN PEDAGÓGICA
# -----------------------------
st.subheader("📘 Explicación")

st.markdown("""
- **KNN**: Clasifica según los vecinos más cercanos.
- **SVM**: Encuentra la mejor frontera de separación.
- **Árbol de Decisión**: Divide los datos en reglas tipo "si-entonces".

Puedes cambiar los parámetros en la barra lateral para ver cómo afecta el modelo.
""")
