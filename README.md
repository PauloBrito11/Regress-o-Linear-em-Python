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

O objetivo deste projeto é prever quantas calorias são queimadas em relação à duração do treino (em horas). 

### Variáveis Utilizadas

| Variável               | Descrição                        | Unidade               |
|------------------------|----------------------------------|-----------------------|
| **Duração do treino (X)** | Duração de cada sessão de treino | Horas por sessão      |
| **Calorias queimadas (Y)** | Calorias queimadas em cada sessão | Calorias por sessão   |

---

### Etapas do Processo

1. **Atribuição de valores às variáveis**  
   Definir os valores de entrada (duração do treino) e os valores de saída (calorias queimadas) para treinamento do modelo.

2. **Criação do modelo e preenchimento com os dados**  
   Construir o modelo de regressão linear e fornecer os dados necessários para que ele aprenda a relação entre a duração do treino e as calorias queimadas.

3. **Verificação dos resultados do modelo**  
   Avaliar o desempenho do modelo ajustado com métricas apropriadas para garantir a precisão das previsões.

4. **Realização de previsões**  
   Usar o modelo treinado para prever quantas calorias são queimadas com base em novas durações de treino.

---

### Etapa 1

```python
#Atribuindo valores

x_values = df[['Session_Duration (hours)']]
y_values = df[['Calories_Burned']]
```

### Etapa 2

```python
#Criando o modelo e preenchendo com os dados

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
#Realizando previsão

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

Lembre-se que todas essas são variáveis INDEPENDENTES e serão utilizadas para prever a variável dependente (''Calories_Burned''). 

---

O processo segue as seguintes etapas:

1. **Atribuição de valores às variáveis**  
   Definir os valores para as variáveis independentes e a variável dependente (`Calories_Burned`), que será o alvo da previsão.

2. **Criação e preenchimento do modelo com os dados**  
   Construir o modelo de regressão e preencher com os dados para que ele aprenda a relação entre as variáveis independentes e a variável dependente.

3. **Realizar previsões**  
   Utilizar o modelo treinado para prever o valor de `Calories_Burned` com base em novas entradas das variáveis independentes.

4. **Verificar a precisão do modelo**  
   Avaliar a precisão das previsões feitas pelo modelo, garantindo que ele produza resultados confiáveis.

---

### Etapa 1

```python
#Atribuindo os valores

x = df.drop(['Calories_Burned', 'Gender', 'Workout_Type'], axis = 1)
y = df['Calories_Burned'].values.reshape(-1,1)
```

### Etapa 2

```python
#Criando o modelo

rl_mult = ln.LinearRegression()
rl_mult.fit(x,y)
```

### Etapa 3

```python
#Realizando previsão

rl_mult.predict(np.array([[46, 80, 1.80, 160, 150, 150, 1.50, 12.6, 2.1, 4, 3, 3]]))
```

### Etapa 4


```python
#Verificando o desempenho do modelo

X = df.drop(['Calories_Burned', 'Gender', 'Workout_Type'], axis = 1)
y = df['Calories_Burned']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
```

![image](https://github.com/user-attachments/assets/6283e0a8-8e36-4352-b706-915b4b14c844)



