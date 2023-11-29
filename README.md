# Databricks

## Engenharia de Dados com Databricks: Extração, Tratamento e Análise de Dados

"Spotify.ipynb"
Exploração e tratamento da base de dados do Spotify

"Recomendador_Musica.ipynb"
Criando um sistema de recomendação de Playlist, baseado nas características de uma música ouvida pelo usuário.

## OBS.: gráficos plotados
Os gráficos não são mostrados nos arquivos, pois foram exportados de notebooks do databricks community. Por isso deixarei as imagens dos gráficos obtidos na análise de dados abaixo:

## Gráficos Spotify.ipynb

# Quantidade de músicas lançadas ao longo do tempo (ano):
![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/d3f806d9-015c-4f37-a0a2-da97132bded1)

# Quantidade de músicas lançadas ao longo do tempo (década):
![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/d72becb1-964e-4388-b688-8a9bb2ea2f3e)

# Média de duração das músicas ao longo do tempo (ano):
![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/64dfa875-97cd-4004-9a40-8fbc27983b7f)

# Comparação de diferentes características de músicas ao longo do tempo (ano):
![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/2ec58c78-0e0c-4118-a8d1-863542652022)

# Comparação de diferentes características de músicas ao longo do tempo (década):
![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/26929927-b8f4-4cf2-b311-d2be158a2dfe)

## Gráficos Recomendador_Musica.ipynb

# Divisão das músicas através dos clusters pca:
fig = px.scatter(projection_kmeans.toPandas(), x='x', y='y', color='cluster_pca', hover_data=['artists_song'])
fig.show()

![image](https://github.com/victorsa2/spotify_analytics/assets/141345545/4f75153c-cfac-4919-a084-44b2a1b57d11)
