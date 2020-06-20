#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    women_users = black_friday.loc[black_friday["Gender"] == "F"]
    women_requested_age = women_users.loc[black_friday["Age"] == "26-35"]
    return women_requested_age.shape[0]
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    unic_users = black_friday["User_ID"].nunique()
    return unic_users


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    unic_dtypes = black_friday.dtypes.nunique()
    return unic_dtypes


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    #Count how many null values are in every row
    rows_nan_count = black_friday.isna().sum(axis = 1)
    
    #Count how many rows have null values
    rows_nan_total = np.count_nonzero(rows_nan_count)

    #Count how many rows have null values in percentage
    nan_percentage = rows_nan_total / black_friday.shape[0]
    return nan_percentage


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    return black_friday.isna().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    values_frequency = black_friday["Product_Category_3"].value_counts()
    return values_frequency.first_valid_index()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    data = np.array(black_friday["Purchase"]).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    
    black_friday["Purchase"] = data_scaled
    return black_friday["Purchase"].mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    data = np.array(black_friday["Purchase"]).reshape(-1, 1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    black_friday["Purchase"] = data_scaled
    return int(black_friday["Purchase"].between(-1, 1, inclusive = False).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    categories = black_friday[["Product_Category_3", "Product_Category_2"]]
    
    #Searching for the rows where the values of "Product_Category_2" aren't a number
    category_na = categories[categories["Product_Category_2"].isna()]
    
    #Comparing the rows where the values of "Product_Category_2" aren't a number with the values of
    #column "Product_Category_3" in the same rows
    return category_na["Product_Category_2"].equals(category_na["Product_Category_3"])


# In[ ]:




