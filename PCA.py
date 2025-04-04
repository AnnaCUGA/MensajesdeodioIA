import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer

with open("CSV/Odioesptrabajado.csv", "rb") as f:
    encoding = chardet.detect(f.read())["encoding"]

df = pd.read_csv('CSV/Odioesptrabajado.csv',encoding=encoding)

df = df.dropna(subset=['text_preprocessed'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_preprocessed'])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['target'], palette='Set1')
plt.title('Diagrama de dispersión con PCA (base descargada)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Target')
plt.show()