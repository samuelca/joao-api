import psycopg2
from psycopg2 import Error
from decouple import config

# *************** Parametros para conexão no Banco de Dados ****************
usuario = config('USUARIO')
senha = config('SENHA')
servidor = config('SERVIDOR')
porta = config('PORTA')
banco_de_dados = config('BANCO_DE_DADOS')



# Função que acessa o banco e retorna as noticias checadas como uma lista
# Cada linha do banco é uma notícia, as colunas são: - Noticia checada
#                                                    - Link da checagem
#                                                    - Data que notícia foi checada
#                                                    - Agência que checou a notícia
#                                                    - Vetores do embedding FastText
#                                                    - Vetores do embedding Flair
def pegar_noticias():
	connection = psycopg2.connect(user = usuario,
									  password = senha,
									  host = servidor,
									  port = porta,
									  database = banco_de_dados)
	cursor = connection.cursor()
	pegar_dados = "SELECT * FROM noticias_embed"
	try:
		cursor.execute(pegar_dados)
		noticias = cursor.fetchall()

		return noticias

	except (Exception, psycopg2.Error) as error :
		print(error)
		return 0