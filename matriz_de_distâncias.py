# Paula Beatriz Louback Jardim

# ------------------------------------------------------
### Enunciado ###

# Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre
# todos os vetores que encontramos usando o tf-idf.
# Para isso use a seguinte fórmula para o cálculo do cosseno use a fórmula apresentada em Word2Vector
# (frankalcantara.com)(https://frankalcantara.com/Aulas/Nlp/out/Aula4.html#/0/4/2)

# O resultado deste trabalho será uma matriz que relaciona cada um dos vetores já calculados 
# com todos os outros vetores disponíveis na matriz termo-documento.

# ------------------------------------------------------
# Realizando os imports:
#!python -m spacy download en_core_web_sm

from bs4 import BeautifulSoup
from requests import get
from spacy import load
import numpy as np
import string
from sklearn.preprocessing import normalize

# ------------------------------------------------------
# Guardando as referências para todos os artigos em uma lista:

artigos = ["https://aliz.ai/en/blog/natural-language-processing-a-short-introduction-to-get-you-started/",
           "https://medium.com/nlplanet/two-minutes-nlp-python-regular-expressions-cheatsheet-d880e95bb468",
           "https://hbr.org/2022/04/the-power-of-natural-language-processing",
           "https://www.activestate.com/blog/how-to-do-text-summarization-with-python/",
           "https://towardsdatascience.com/multilingual-nlp-get-started-with-the-paws-x-dataset-in-5-minutes-or-less-45a70921d709"]

# ------------------------------------------------------
# Esse código percorre todos os documentos, criando uma lista com as sentenças de cada documento
# e ao final, une todas essas listas em uma matriz.
# Ao mesmo tempo, o código abaixo realiza a mesma atividade descrita acima para o vocabulário de cada documento,
# criando uma lista de palavras para cada um e juntando essas listas no final. 

matriz_palavras = [[], [], [], [], []]
matriz_sentencas = []

i = 0
for site in artigos:
  sents_list = []
  document_words = set()

  r = get(site)
  r = r.content

  soup = BeautifulSoup(r, 'html.parser')
  text = soup.find_all('p')
  nlp = load("en_core_web_sm")

  for paragraph in text:
    content = paragraph.get_text()
    sentences = nlp(content).sents

    for sent in sentences:
      sent = sent.text.strip(string.punctuation)
      sent = sent.strip(string.digits)
      sent = sent.strip('\n')
      sents_list.append(sent)
      words = sent.split(" ")
      
      for word in words:
        word = word.strip(string.punctuation)
        word = word.strip(string.digits)
        word = word.strip('\n')
        document_words.add(word)

  matriz_sentencas.append(sents_list)
  for w in document_words:
    matriz_palavras[i].append(w)

  i += 1

# ------------------------------------------------------
# O trecho abaixo cria um Bag of words unindo todas as palavras de todos os documentos.
# Neste trabalho estou considerando uppercases e lowcases como caracteres diferentes.

corpus = set()

for lists in matriz_palavras:
  for word in lists:
    corpus.add(word)

# ------------------------------------------------------
# Logo abaixo está a criação da header da matriz-termo.

header = []

for each in corpus:
  if each != "":
    header.append(each)

header = sorted(header)

# ------------------------------------------------------
# Agora, vamos criar a matriz-termo.
# Essa matriz está sendo criada em um dicionário, onde cada sentença é uma key e o valor de cada key é uma lista
# na qual os elementos correspondem a quantidade de vezes que cada termo da header aparece nessa sentença.

dict_matriz= {}
dict_matriz["Sentenças"] = header[1:]

for doc in matriz_sentencas:
  for sent in doc:
    values_list = []
    termos = sent.split(" ")
    for palavra in dict_matriz["Sentenças"]:
      contador = 0
      for cada_plv in termos:
        if cada_plv == palavra:
          contador += 1
      values_list.append(contador)
    dict_matriz[sent] = values_list

# ------------------------------------------------------
# No trecho abaixo, estou criando uma lista que armazena a quantidade de sentenças em que cada termo aparece.
# Essa lista vai ser usada na sequência para calcular o IDF.

qnt_sent = np.zeros(len(dict_matriz["Sentenças"]), dtype=float)
qnt_sent = list(qnt_sent)

for sent in dict_matriz:
  if sent != "Sentenças":
    index = 0
    while index < len(dict_matriz[sent]):
      valor = int(dict_matriz[sent][index])
      if valor != 0:
        qnt_sent[index] += 1  
      index += 1

# ------------------------------------------------------
# Agora, vou realizar o cálculo do TF-IDF para todos os termos em cada sentença
# e então substituir esses valores no dicionário onde está minha matriz.

for sent in dict_matriz:
  if sent != "Sentenças":
    termos = sent.split(' ')
    qnt_termos = len(termos) # quantidade de termos na sentença atual

    tfidf = np.zeros(len(dict_matriz["Sentenças"]), dtype=float)
    index = 0
    while index < len(dict_matriz[sent]):
      valor = dict_matriz[sent][index]
      if qnt_sent[index] != 0:
        calculo = (valor/qnt_termos) * np.log((len(dict_matriz) - 1)/qnt_sent[index])
        tfidf[index] = calculo
      index += 1
    tfidf_list = normalize([tfidf]) # normaliza os valores para ficar entre 0 e 1
    dict_matriz[sent] = list(tfidf_list[0])

# ------------------------------------------------------
# Abaixo estou setando os valores pra no máximo 3 casas decimais

for sent in dict_matriz:
  if sent != "Sentenças":
    index = 0
    while index < len(dict_matriz[sent]):
      valor = dict_matriz[sent][index]
      new_valor = float(str(valor)[:5])
      dict_matriz[sent][index] = new_valor
      index += 1

# ------------------------------------------------------
# Abaixo crio uma matriz apenas com os vetores que apresentam os valores da TF-IDF para cada sentença,
# pois é mais facil trabalhar com lista do que com dicionário,
# em seguida vou armazenar o resultado da matriz de distâncias no dicionário que eu já havia criado

list_matriz = []

for sent in dict_matriz: 
  if sent != "Sentenças":
    list_matriz.append(dict_matriz[sent])

# ------------------------------------------------------
# aqui estou criando uma matriz de mesmas proporções da matriz anterior, porém, contendo apenas 1.0
# não quis usar a função np.ones pois ela criaria um numpy.ndarray e eu teria que transformar esse array em lista.

matriz_distancias = []

for vetor in range(0, len(list_matriz)):
  matriz_distancias.append([])
  index = 0
  while index < len(list_matriz[0]):
    matriz_distancias[vetor].append(1.0)
    index += 1

# ------------------------------------------------------
# Abaixo estou fazendo o cálculo do cosseno e atribuindo o valor à sua devida posição nos vetores da matriz de distâncias.

for i in range(0, len(list_matriz)):
  for j in range(1, len(list_matriz) - 1):
    cos = np.dot(list_matriz[i], list_matriz[j])/(np.linalg.norm(list_matriz[i]) * np.linalg.norm(list_matriz[j]))
    matriz_distancias[i][j] = cos
    matriz_distancias[j][i] = cos

# ------------------------------------------------------
# Agora que já calculei as distâncias, vou levar os vetores com os resultados para um dicionário.

lista_sentencas = []
dict_distancias = {}

for sent in dict_matriz:
  if sent != "Sentenças":
    lista_sentencas.append(sent)

for sent, vetor in zip(lista_sentencas, matriz_distancias):
  dict_distancias[sent] = vetor

# ------------------------------------------------------
# Como o Colab limita o output por conta do tamanho do dicionario, abaixo estou printando apenas as 25 primeiras linhas da matriz.
# Acho que o código tem algum bug que não consegui descobrir, por isso alguns valores estão como dízimas periódicas.

print(dict_matriz["Sentenças"])

loop = 0
for sent in dict_distancias:
  if loop <= 25:
    print(dict_distancias[sent])
  loop += 1