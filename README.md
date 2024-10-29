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

1. Atribuição de valores as variaveis
2. Criação do modelo e preenchimento com os dados
3. Verificar os resultados do modelo
4. Realizar previsões

### Etapa 1

```python
x_values = df[['Session_Duration (hours)']]
y_values = df[['Calories_Burned']]
```

### Etapa 2

```python
modelo = ln.LinearRegression()
modelo.fit(x_values, y_values)
```

### Etapa 3

```python
#Verificando o resultado

print('Utilizando biblioteca Scikit-Learn')
print('y = ax + b')
print('a = %.2f => a inclinação da linha de tendência.' % modelo.coef_[0][0])
print('b = %.2f => o ponto onde a linha de tendência atinge o eixo y.' % modelo.intercept_[0])

```

```python
#Verificando graficamente

y_previsao = modelo.predict(x_values)

plt.figure(1, figsize = (10,8))
plt.scatter(df['Session_Duration (hours)'], df['Calories_Burned'], label = 'Dados reais')
plt.plot(df['Session_Duration (hours)'], y_previsao, color = 'red', label = 'linha de tendencia')
plt.legend()
plt.show()
```
#### Visualização gráfica
![image](https://github.com/user-attachments/assets/745cfb7b-5ff6-4103-8a42-018d1b8c374a)

### Etapa 4

```python
valor_p1 = 0.8
previsao = modelo.predict(np.array([[valor_p1]]))

print(f'Previsão: {previsao}')
#Resultado: [[575.98235389]]
```

## Regressão linear múltipla

O objetivo continua o mesmo, prever quantas calorias são queimadas, porém, dessa vez, teremos mais variáveis na equação, são elas:

| Variável (Xs)                 | Descrição                                                                                     | Unidade                |
|---------------------------|-----------------------------------------------------------------------------------------------|------------------------|
| Age                       | Idade do indivíduo                                                                            | Anos                   |
| Weight (kg)               | Peso do indivíduo                                                                             | Quilogramas (kg)       |
| Height (m)                | Altura do indivíduo                                                                           | Metros (m)             |
| Max_BPM                   | Frequência cardíaca máxima durante o treino                                                   | Batimentos por minuto  |
| Avg_BPM                   | Frequência cardíaca média durante o treino                                                    | Batimentos por minuto  |
| Resting_BPM               | Frequência cardíaca em repouso                                                                | Batimentos por minuto  |
| Session_Duration (hours)  | Duração de cada sessão de treino                                                              | Horas                  |
| Fat_Percentage            | Percentual de gordura corporal do indivíduo                                                   | Porcentagem (%)        |
| Water_Intake (liters)     | Consumo de água do indivíduo por dia                                                          | Litros (L)             |
| Workout_Frequency (days/week) | Frequência semanal de sessões de treino                                                   | Dias por semana        |
| Experience_Level          | Nível de experiência do indivíduo com treinos                                                | Escala (Ex: 1-10)      |
| BMI                       | Índice de Massa Corporal, calculado como peso dividido pela altura ao quadrado                | kg/m²                  |

Lembre-se que todas essas são variáveis INDEPENDENTES e serão utilizadas para prever a variável dependente (''Calories_Burned''). O processo segue as seguintes etapas:

1. Atribuição de valores às variáveis
2. Criação do modelo e preenchimento do modelo com os dados
3. Realizar previsões
4. Verificar a precisão do modelo

### Etapa 1

### Etapa 2

### Etapa 3

### Etapa 4

