MLP1 
	- 20sec
		- Acurácia 100%
		- Estabilizou após 25 épocas
	- trim5
		- Acurácia 88%
		- Estabilizou após 40 épocas
		- No final, acc diminuiu
	- trim10
		- Acurácia 86%
		- Estabilizou após 40 épocas
	- slide5
		- Acurácia 91%
		- Estabilizou após 60 épocas
	- slide10
		- Acurácia 94%
		- Estabilizou após 75 épocas
	- slide10phone
		- Acurácia 97%
		- Estabilizou após 60 épocas
	- slidesr22050
		- Acurácia 93%
		- Estabilizou após 40 épocas
	- Dropout
		- Diminuiu o desempenho. Não foi notado necessidade de dropout, mas deve-se observar que o conjunto de teste pode compartilhar pacientes com o conjunto de treinamento.

CNN1
	- 20sec
		- Primeiro treinamento: resultado médio
		- Segundo treinamento: parecido
		- Acurácia 73%
	- trim5
		- Acurácia 97%
		- Saturacao no final, aprendizado mais "lento"
	- trim 10 
		- Meio termo 20sec e trim5
		- Acurácia 73%
		- Confundiu pneumonia com saudável
	- slide5
		- Acurácia 99%
		- Cometeu 1 erro: confundiu pneumonia com saudável
	- slide10
		- Acurácia 100%
		- Menor variacao gráfico validacao
	- slide10phone
		- 100%
		- variacao validacao um pouco maior que slide10
	- slide10sr22050
		- 100%
		- pouca variacao grafico validacao

CNN2
	- 20sec
		- muito parecido com cnn1
		- Acurácia 66%
	- trim5
		- Acurácia 88%
	- trim 10 
		- 80%
	- slide5
		- Acurácia 98%
		- Cometeu 2 erros: confundiu pneumonia com saudável
	- slide10
		- Acurácia 100%
		- Menor variacao gráfico validacao
	- slide10phone
		- 98%
	- slide10sr22050
		- 100%
	

Observacoes:
	- Retreinar produz resultados diferentes em alguns casos.
	- CNN se dá bem com slide10, resultados promissores.
	- MLP parece ter um comportamento mais variável nas primeiras épocas, e uma estabilizacao até o final.
	- MLP satura rápido, parece que a rede está no limite:
		- Se aumentar dropout ou diminuir neurônios o desempenho diminui.
	- Pouca diferenca entre tempo.
	- Em alguns treinamentos MLP, um fato interessante: a matriz de confusao revela que os modelos estão mais propensos a classificar o saudável como doente, do que o contrário. 

	Com relacao as dados:
		- o dataset apresenta anotacoes de certos sons produzidos em cada ciclo da respiracao. Esses sons ajudam ao médico na identificacao da patologia. Pode ser interessante utilizar estes dados.
		- POde ser interessante criar um conjunto de teste. 2 pacientes de cada classe?
		- Cuidado resultado teste slide: resultado pode ser enganoso já que as instâncias se repetem mais ainda.


Sugestões:
	- Criar um conjunto de teste apropriado e retreinar?
	- Testar topologias do vitor?
	- Qual o tempo necessário? Alguns treinamentos sugerem que o tempo menor é melhor (MLP) mas minha intuicao é que não. Testar?
	- Em contrapartida, slide10 parece ter resultado promissor em todos, principalmente CNN, mas o teste pode ser que seja mais falho. Refazer com um conjunto de teste apropriado?
	- Esse resultado é satisfatório? Como podemos verificar se o modelo realmente aprendeu e pode ser interessante?


Outros pontos:
	- Análise dos dados
	- Dataset contém inforamćões interessantes de cada ciclo de respiracao


Tarefas futuras
	1. Treinar novamente com conjunto de teste;
	2. Treinar com modelos do Vitor;
	3. Treinar topologias small data;
