import pandas as pd
from colorama import Fore
import chardet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Cargar los datos nuevamente, para tener los datos actualizados y sin problemas
with open("CSV/Odioesptrabajado.csv", "rb") as f:
    encoding = chardet.detect(f.read())["encoding"]
df = pd.read_csv('CSV/Odioesptrabajado.csv', encoding=encoding)

#Eliminar posibles valores nulos en la columna que nos interesa clasificar
df = df.dropna(subset=['text_preprocessed'])

#Preprocesamiento del texto: eliminar palabras "co" y "http" en la columna de text_preprocessed, estas palabras deben ser retiradas por las razones que vimos arriba
#puede ser que se eliminen otras
df['text_preprocessed'] = df['text_preprocessed'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in ('co', 'http')]))

#Vectorización de texto utilizando TF-IDF nuevamente para el valor de las características, la columna text_preprocessed
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_preprocessed'])
#definimos el objetivo, la columna target
y = df['target']

#Dividir los datos en conjuntos de entrenamiento y prueba, tomando 80 de train y 20 de test, con semilla 170119
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=170119)

#crear el modelo de clasificación con RandomForest utilizando los mejores hiperparámetros, es decir, el mejor modelo
best_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=170119)

#Entrenar el modelo
best_model.fit(X_train, y_train)

#Realizar predicciones en los datos de prueba a partir del modelo
y_pred = best_model.predict(X_test)


#Calculamos las métricas de evaluación del modelo, usando average weighted
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

#Mostrar métricas de evaluación
#Metricas de evaluación para el modelo de random forest
print(Fore.BLUE + "Metricas de evaluación para el modelo de Random Forest")
print(Fore.GREEN + "\nAccuracy:", accuracy)
print(Fore.CYAN + "\nPrecision:", precision)
print(Fore.MAGENTA + "\nRecall:", recall)
print(Fore.YELLOW + "\nF1 Score:", f1)

#Calculando la accuracy con 5 de crossvalidation a ver si mejora
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

#Accuracy para el método de random forest con 5 de crossvalidation
print(Fore.WHITE + "\nAccuracy para el modelo de Random Forest con 5 de Cross Validation")
print(Fore.RED + "\nAccuracy con validación cruzada (5-fold):", cv_scores.mean())

#Además, vamos a mostrar la eficiencia de clasificación con la métrica de precisión para cada una de las categorías
#Calcular la precisión por clase
precisionporclase = precision_score(y_test, y_pred, average=None)

#tomamos los nombres de las diferentes categorías usando la función unique, esta es para quitar duplicados, pero se puede usar para esto
categorias = df['target'].unique()
for categoria in categorias:
    print(Fore.GREEN + categoria)

#Mostrar la precisión para cada una de las clases
print(Fore.BLUE + "\nPrecisión por clase:")
#iteramos en cada una de las precisiones que se calcularon, un total de 6
for i in range(len(precisionporclase)):
#imprimimos el nombre de la categoría además de la precisión obtenida
    print(Fore.GREEN + f"{categorias[i]}: {precisionporclase[i]}")

# Generar la curva de aprendizaje
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Tamaño del Conjunto de Entrenamiento")
    plt.ylabel("Exactitud")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación")

    plt.legend(loc="best")
    return plt

plot_learning_curve(best_model, "Curva de Aprendizaje (Random Forest)", X, y, cv=5)
plt.savefig("images/Graficacincuenta.png")
plt.show()