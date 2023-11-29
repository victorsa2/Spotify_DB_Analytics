# Databricks notebook source
import pyspark.pandas as ps

# COMMAND ----------

path = 'dbfs:/FileStore/tables/spotfy/dados_tratados/data.parquet'
df_data = ps.read_parquet(path)

# COMMAND ----------

df_data.info()

# COMMAND ----------

df_data = df_data.dropna()

# COMMAND ----------

df_data.info()

# COMMAND ----------

df_data['artists_song'] = df_data.artists + ' - ' + df_data.name

# COMMAND ----------

df_data.head()

# COMMAND ----------

df_data.info()

# COMMAND ----------

X = df_data.columns.to_list() #Tranformando os nomes das colunas em uma lista e removendo elas da lista
X.remove('artists')
X.remove('id')
X.remove('name')
X.remove('artists_song')
X.remove('release_date')
X

# COMMAND ----------

# MAGIC %md
# MAGIC #Transformando o data framepandas no dataframe sql do spark
# MAGIC
# MAGIC O pandas não suporta um modelo de mli, então passamos para o spark

# COMMAND ----------

df_data = df_data.to_spark()

# COMMAND ----------

#Temos que tranformar os dados colunares em vetores (lista), porque é como o MLI funciona

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

dados_encoded_vector = VectorAssembler(inputCols=X,
outputCol='features').transform(df_data)

# COMMAND ----------

dados_encoded_vector.select('features').show(truncate=False, n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC #Padrionizando os dados

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
model_scaler = scaler.fit(dados_encoded_vector)
dados_musicas_scaler = model_scaler.transform(dados_encoded_vector)

# COMMAND ----------

dados_musicas_scaler.select('features_scaled').show(truncate=False, n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC #PCA
# MAGIC Reduzir o número de features por PCA

# COMMAND ----------

k = len(X)
k

# COMMAND ----------

from pyspark.ml.feature import PCA

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

sum(model_pca.explainedVariance) * 100

# COMMAND ----------

lista_valores = [sum(model_pca.explainedVariance[0:i+1]) for i in range(k)]
lista_valores

# COMMAND ----------

import numpy as np

# COMMAND ----------

k = sum(np.array(lista_valores) <= 0.7)
k

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca_final = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

dados_musicas_pca_final.select('pca_features').show(truncate=False, n=5)

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

pca_pipeline = Pipeline(stages=[VectorAssembler(inputCols=X, outputCol='features'),
                                StandardScaler(inputCol='features', outputCol='features_scaled'),
                                PCA(k=6, inputCol='features_scaled', outputCol='pca_features')])

# COMMAND ----------

model_pca_pipeline = pca_pipeline.fit(df_data)

# COMMAND ----------

projection = model_pca_pipeline.transform(df_data)

# COMMAND ----------

projection.select('pca_features').show(truncate=False, n=5)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

SEED = 1224

# COMMAND ----------

kmeans = KMeans(k=50, featuresCol='pca_features', predictionCol='cluster_pca', seed=SEED)

# COMMAND ----------

modelo_kmeans = kmeans.fit(projection)

# COMMAND ----------

projection_kmeans = modelo_kmeans.transform(projection)

# COMMAND ----------

projection_kmeans.select(['pca_features', 'cluster_pca']).show()

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

# COMMAND ----------

projection_kmeans = projection_kmeans.withColumn('x', vector_to_array('pca_features')[0])\
                                   .withColumn('y', vector_to_array('pca_features')[1])

# COMMAND ----------

projection_kmeans.select(['x', 'y', 'cluster_pca', 'artists_song']).show()

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

fig = px.scatter(projection_kmeans.toPandas(), x='x', y='y', color='cluster_pca', hover_data=['artists_song'])
fig.show()

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'

# COMMAND ----------

cluster = projection_kmeans.filter(projection_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
cluster

# COMMAND ----------

musicas_recomendadas = projection_kmeans.filter(projection_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
musicas_recomendadas.show()

# COMMAND ----------

componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]
componenetes_musica                             

# COMMAND ----------

from scipy.spatial.distance import euclidean
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f

# COMMAND ----------

def calcula_distance(value):
    return euclidean(componenetes_musica, value)

udf_calcula_distancia = f.udf(calcula_distance, FloatType())

musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distancia('pca_features'))

# COMMAND ----------

recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

recomendadas.show()

# COMMAND ----------

def recomendador(nome_musica):
    cluster = projection_kmeans.filter(projection_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projection_kmeans.filter(projection_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]

    def calcula_distance(value):
        return euclidean(componenetes_musica, value)

    udf_calcula_distance = f.udf(calcula_distance, FloatType())

    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

    recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

    return recomendadas

# COMMAND ----------

df_recomedada = recomendador('Taylor Swift - Blank Space')
df_recomedada.show()

# COMMAND ----------


