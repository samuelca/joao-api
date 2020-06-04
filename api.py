# ************ Imports ****************
import pandas as pd 
import os

from flair.embeddings import DocumentPoolEmbeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

import numpy as np 
from api_banco import pegar_noticias

# ********************** Recuperação das notícias já convertidas em vetores ******************************

# Cada notícia é convertida em dois vetores de características usando dois tipo de embeddings, fastText e Flair.
# Os vetores convertidos pelo FastText possuem 300 entradas cada um, já os convertidos pelo Flair possuem 4095 entradas
# Esses vetores foram salvos em um arquivo CSV para evitar que se tenha que converte-los sempre que o código iniciar
# Essa parte do código faz a recuperação desses vetores na pasta "static".

# Quando importados, os vetores vem como uma string, essa função converte a string em um vetor de floats
def converte_float(vetor):
	v = vetor[1:][:-1].split(",")
	lista = []
	for i in range(0,len(v)):
		lista.append(float(v[i]))
	return lista


db = pegar_noticias()
df = pd.DataFrame(db)
''' O DataFrame possui 6 colunas: 0 -> Texto da Notícia
                                  1 -> Link da Notícia
                                  2 -> Data da Checagem
                                  3 -> Agencia que realizou a checagem
                                  4 -> vetores do embedding FasText
                                  5 -> Vetores do embedding Flair
'''
df[4] = df[4].apply(converte_float) # Vetores fast convertidos de String para Float
df[5] = df[5].apply(converte_float) # Vetores flair convertidos de String para Float

vetores_flair = df.drop(columns = [1,2,3,4]) # Deixando somente a coluna com a noticia e a coluna dos vetores Flair
vetores_fast = df.drop(columns = [1,2,3,5]) # Deixando somente a coluna com a noticia e a coluna dos vetores Fast

#vetores_flair = pd.read_csv(os.path.abspath("static/vetor_flair.csv")).drop(columns = "Unnamed: 0")
#vetores_fast = pd.read_csv(os.path.abspath("static/vetor_fastT.csv")).drop(columns = "Unnamed: 0")
#vetores_flair['1'] = vetores_flair['1'].apply(converte_float)
#vetores_fast['1'] = vetores_fast['1'].apply(converte_float)

# Convertendo os vetores que estão na forma de DataFrame para lista. Essa conversão facilita na iteração posterior
flair = vetores_flair.values.tolist()
fastT = vetores_fast.values.tolist()

dicio_vetores = {"fastT" : fastT, "flair" : flair}

# **************************** Definição dos Embeddings ********************************************

# Aqui é inicializado o embeddings do FastText em português
# A opreação seguinte define o embedding para documentos, usando o método Pool para agregar cada embeddings das palavras
pt_embedding = WordEmbeddings('pt')
document_embedding = DocumentPoolEmbeddings([pt_embedding])

# Inicializando os embeddings do Flair
flair_embedding_forward = FlairEmbeddings('pt-forward')
flair_embedding_backward = FlairEmbeddings('pt-backward')

# Para o Flair é recomendado inicializar dois tipos de embeddings, forward e backward, e empilha-los usando StackedEmbeddings
stacked_embeddings = StackedEmbeddings([
										flair_embedding_forward,
										flair_embedding_backward,
									   ])

document_embedding_flair = DocumentPoolEmbeddings([stacked_embeddings])

embeddings = {"fastT": document_embedding, "flair":document_embedding_flair}

# ************** Cálculo de similaridade entre as notícias *********************************

# Para descobrir qual a notícia mais similar do banco com a frase que foi recebida, primeiro a frase é convertida em vetor
# usando os embaddings, em seguida para cada um dos embaddings é feito o seguinte procedimento:
# 	1) O vetor da frase (v1) é comparado com cada um dos vetores do banco (v2) para que se descubra qual o mais similar,
# a medida de simimilaridade usada é o arco-cosseno dado pela fórmula arcos(<|v1|,|v2|> / (|v1|*|v2|))
# 	2) Após feito esse cálculo para todas as notícias do banco, a lista com os valores encontrados para a similaridade é ordenada
# e são retornados as 10 noticias com maior similaridade
#	3) Serão selecionadas como retorno da API aquelas noticias que estiverem presente em ambas as listas de 10 mais similares
  
def arcos(vet1, vet2):
	vetor_np1 = vet1.detach().numpy()
	vetor_np2 = vet2
	simi = (np.dot(vetor_np1,vetor_np2)) / (np.linalg.norm(vetor_np1) * np.linalg.norm(vetor_np2))
	return np.arccos(simi)

def similaridade(frase, tipo_embeding):

	sentence = Sentence(frase, use_tokenizer=True)
	embeddings[tipo_embeding].embed(sentence)

	vetor_frase = sentence.get_embedding()

	lista_similaridade = []
	for i in dicio_vetores[tipo_embeding]: # Primeira posição do vetor é a frase e a segunda o vetor de embedings
		lista_similaridade.append((arcos(vetor_frase,i[1]), i[0]))

	lista_similaridade.sort(reverse = False)

	return lista_similaridade[:10]

def gerar_respostas(frase):
	lista_embedings = ['fastT','flair']
	lista_similares = []
	for emb in lista_embedings:
		lista_similares.append(similaridade(frase,emb))
	
	lista_final = []

	for i in range(0,len(lista_similares[0])):
		noticia_fast = lista_similares[0][i]
		for j in range(0,len(lista_similares[0])):
			noticia_flair = lista_similares[1][j]
			if (noticia_fast[1] == noticia_flair[1]):
				lista_final.append(noticia_fast)
	
	return lista_final
