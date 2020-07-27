#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


countries = pd.read_csv("countries.csv")


# In[3]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[4]:


countries["Country"] = countries["Country"].apply(lambda x: x.strip())
countries["Region"] = countries["Region"].apply(lambda x: x.strip()) 


# In[5]:


countries.replace(',', '.', regex=True, inplace=True)
countries.iloc[:, 2:] = countries.iloc[:, 2:].astype("float64")


# In[6]:


countries.info()


# In[7]:


countries.head()


# In[8]:


countries.describe()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    # Selecting the regions and removing blank spaces
    regions = countries["Region"].apply(lambda x: x.strip()).unique()
    
    # Transforming the numpy.array to list and sorting it alphabetically
    sorted_regions = sorted(regions.tolist())
    
    return sorted_regions


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


def q2():
    # Initializing KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    
    # Discretizing the colum "Pop_density"
    pop_density_discretized = discretizer.fit_transform(countries[["Pop_density"]])
    
    # Returning the sum of the countries above 90% percentile
    return int(sum(pop_density_discretized == 9))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[11]:


def q3():
    # Counting the unique values of the variables "Region" and "Climate"
    num_values_region = countries["Region"].unique().size
    num_values_climate = countries["Climate"].unique().size
    
    # Returning the sum of the unique values from both variables
    return num_values_region + num_values_climate


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[12]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[15]:


def q4():
    # Declaring Pipeline
    pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard", StandardScaler())])
    
    # Fitting data
    pipe.fit(countries.iloc[:, 2:])
    
    # Transforming data
    test_data_transformed = pipe.transform([test_country[2:]])

    # Returning the value referred to the column "Arable" on the transformed data
    return round(test_data_transformed[0, countries.columns.get_loc("Arable") - 2], 3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[114]:


def q5():
    # Declaring the first and third quantile of the variable Net_migration
    first_quantile = countries["Net_migration"].quantile(0.25)
    third_quantile = countries["Net_migration"].quantile(0.75)
    
    # Calculating the IQR
    iqr = third_quantile - first_quantile
    
    # Looking for outliers acoording to boxplot's method
    n_outliers_above = countries["Net_migration"][(countries["Net_migration"] > third_quantile + 1.5 * iqr)].count()
    n_outliers_below = countries["Net_migration"][(countries["Net_migration"] < first_quantile - 1.5 * iqr)].count()
    
    # Returning a tuple with: number of outliers on the inferior side, then on the superior side and lastely,
    # a boolean represeting the answer of the question "these outliers should be removed?"
    return tuple((n_outliers_below, n_outliers_above, False))


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[32]:


def q6():
    # Loading data
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    
    # Vectorizing the count of the words in the data
    count_vectorizer = CountVectorizer()
    newsgroup_vectorized = count_vectorizer.fit_transform(newsgroup.data)
    
    # Counting how many times the word "phone" appears on data
    idx_phone = count_vectorizer.vocabulary_.get('phone')
    phone_counted = newsgroup_vectorized[:, idx_phone].toarray().sum()
    
    return phone_counted


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[34]:


def q7():
    # Loading data
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    
    # Vectorizing the count of the TF-IDF features on data
    tfidf_vectorizer = TfidfVectorizer()
    newsgroup_vectorized = tfidf_vectorizer.fit_transform(newsgroup.data)
    
    # Counting how many times the word "phone" appears on data
    idx_phone = tfidf_vectorizer.vocabulary_.get('phone')
    phone_counted = newsgroup_vectorized[:, idx_phone].toarray().sum()
    
    return round(phone_counted, 3)


# In[ ]:




