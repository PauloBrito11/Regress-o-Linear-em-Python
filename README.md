# Regressão Linear simples e múltipla utilzando Python


Conjunto de dados: kaggle.com/ (adicionar depois)

```python

#Bibliotecas utilizadas:

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as ln

```

## Regressão linear simples 

O objetivo é prever quantas calorias são queimadas em relação a duração do treino (em horas), sendo assim, as variáveis utilizadas serão:

| Variável                      | Descrição                                 | Unidade                   |
|-------------------------------|-------------------------------------------|---------------------------|
| Duração do treino  (X)           | Duração de cada sessão de treino          | Horas por sessão          |
| Calorias queimadas (Y)            | Calorias queimadas em cada sessão         | Calorias por sessão       |

Abaixo estão as etapas do processo e suas funções: 

1. Etapa: Atribuição de valores as variaveis
2. Criação do modelo e preenchimento com os dados
3. Verificar os resultados do modelo
4. Realizar previsões
   

## Regressão linear múltipla
