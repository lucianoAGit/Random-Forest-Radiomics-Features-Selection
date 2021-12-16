# Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# Caminho do arquivo e leitura
csvPath = "Caminho da pasta com o arquivo com as caracteristicas"
df = pd.read_csv(csvPath)
df = df.fillna(0)

# Normalizar os dados com Standard Scale
x = df.drop(['ID'], axis = 1)
x = x.drop(['Grupo'], axis = 1)
column_names = list(x.columns)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)
df_norm = pd.DataFrame(scaled_data, columns=column_names)

# Criacao do modelo
y = df['Grupo']
model = RandomForestClassifier(criterion= 'entropy',n_estimators=1000, bootstrap = True, max_features=10,n_jobs=-1, max_depth= 10)
model.fit(df_norm, y)

# Validação do modelo com 5 folds
scores_dt = cross_val_score(model, df_norm, y, scoring='accuracy', cv=5)
print(scores_dt.mean())

# Score de cada um dos atributos
print(model.feature_importances_)

# Tabela das importancias
feature_importances = pd.DataFrame(model.feature_importances_*100,index = df_norm.columns,columns=['Importância (%)']).sort_values('Importância (%)', ascending=False)

# Selecao de caracteristicas com base em um criterio de peso
df_eq = feature_importances.round(3)
indexNames = df_eq[(df_eq['Importância (%)'] <= 0.4)].index
df_eq.drop(indexNames , inplace=True)
#print(df_eq)

#Salva as caracteristicas selecionadas em um arquivo CSV
df_eq.to_csv("Caminho da pasta para salvar o CSV", index = True)

# Mostra o nome da feature e a porcentagem de importancia
features = column_names
features_importance = zip(model.feature_importances_, features)
for importance, feature in sorted(features_importance, reverse=True):
    print("%s: %f%%" % (feature, importance*100))

# Constroi um grafico de barras com as 30 primeiras caracteristicas
graf = feature_importances.head(30).plot(kind='bar')
graf.figure.savefig('Caminho da pasta para salvar o arquivo de imagem', dpi=300, bbox_inches = "tight")


# Funcoes que podem ser utilizadas adicionalmente:
# Selecao da melhor combinação de parametros para o modelo
param_grid = {
            "criterion": ['entropy', 'gini'],
            "n_estimators": [50, 75, 100],
            "bootstrap": [True],
            "max_depth": [10],
            "max_features": ['auto', 10]
}

grid_search = GridSearchCV(model, param_grid, scoring="accuracy")
grid_search.fit(df_norm, y)

model = grid_search.best_estimator_ 
grid_search.best_params_, grid_search.best_score_

# Heatmap/Matriz de correlação
fig, ax = plt.subplots(figsize=(18,14))
sns.heatmap(df.corr(),ax = ax)