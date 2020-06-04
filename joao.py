# Para executar: export FLASK_APP=joao.py
#                flask run
# ls -A ~
# source env/bin/activate


# Conhecereis a Verdade e a Verdade Vos Libertará João 8, 32

from flask import Flask
from markupsafe import escape
from flask import request
from flask import jsonify



import pandas as pd 
import os

from api import gerar_respostas
from api_banco import pegar_noticias


noticias = pegar_noticias()
df = pd.DataFrame(noticias)
df.drop(columns = [4,5], inplace = True)
df.rename(columns = {0:'noticia', 1:'link', 2:'data', 3:'checagem'}, inplace = True)


app = Flask(__name__)


def checar(frase):
	respostas = gerar_respostas(frase)
	lista = []
	if (respostas):
		for resp in respostas:
			ind = df[df['noticia'] == resp[1]]['data'].index #Gambiarra pra printar direito
			dicio = {'Checado' : resp[1], 'Data' :  df['data'].loc[[i for i in ind][0]],
					'Checado por' : df['checagem'].loc[[i for i in ind][0]],
					'Link' : df['link'].loc[[i for i in ind][0]]}
			lista.append(dicio)
		
	return jsonify(lista)


@app.route('/checagem/<frase>', methods=['GET', 'POST'])
def login(frase):
	if request.method == 'GET':
		return checar(escape(frase))
	else:
		return "Não encontrado"