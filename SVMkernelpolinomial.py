import pandas as pd
import chardet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from colorama import Fore

# Cargar los datos nuevamente, para tener los datos actualizados y sin problemas
with open("CSV/Odioesptrabajado.csv", "rb") as f:
    encoding = chardet.detect(f.read())["encoding"]
df = pd.read_csv('CSV/Odioesptrabajado.csv', encoding=encoding)

# Eliminar posibles valores nulos en la columna que nos interesa clasificar
df = df.dropna(subset=['text_preprocessed'])

# Preprocesamiento del texto: eliminar palabras "co" y "http" en la columna de text_preprocessed
df['text_preprocessed'] = df['text_preprocessed'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in ('co', 'http')]))

# Vectorización de texto utilizando TF-IDF para el valor de las características
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_preprocessed'])
y = df['target']

# Dividir los datos en conjuntos de entrenamiento y prueba, tomando 80 de train y 20 de test, con semilla 170119
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=170119)

# Crear el modelo de clasificación con SVM
svm_model = SVC(kernel='poly', C=1, random_state=170119)

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba a partir del modelo
y_pred = svm_model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Mostrar métricas de evaluación
print(Fore.BLUE + "Métricas de evaluación para el modelo de SVM")
print(Fore.GREEN + "\nAccuracy:", accuracy)
print(Fore.CYAN + "\nPrecision:", precision)
print(Fore.MAGENTA + "\nRecall:", recall)
print(Fore.YELLOW + "\nF1 Score:", f1)

# Calcular la accuracy con 5 de crossvalidation
cv_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

# Accuracy para el método SVM con 5 de crossvalidation
print(Fore.WHITE + "\nAccuracy para el modelo de SVM con 5 de Cross Validation")
print(Fore.RED + "\nAccuracy con validación cruzada (5-fold):", cv_scores.mean())

# Calcular la precisión por clase
precision_por_clase = precision_score(y_test, y_pred, average=None)

# Mostrar la precisión para cada una de las clases
categorias = df['target'].unique()
for categoria in categorias:
    print(Fore.GREEN + categoria)

print(Fore.BLUE + "\nPrecisión por clase:")
for i in range(len(precision_por_clase)):
    print(Fore.GREEN + f"{categorias[i]}: {precision_por_clase[i]}")