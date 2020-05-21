# Experimento 3

Com base no experimento 2, percebemos a necessidade de balancear e maximizar os dados de treinamento para combater o overffiting e também a tendência em classificar nova instâncias como "Healthy". Também percebemos que os modelos com uma duracão um pouco maior obtiveram resultados mais promissores.
A partir do experimento 2, selecionamos os melhores modelos, criamos alguns datasets de 8 segundos e balanceamos os dados antes do treinamento. Para maximizar o conjunto de treinamento, utilizamos os dados de teste para validacão.

Os datasets novos foram criados com a duracão de 8 segundos para garantir as técnicas de trimming e preservar uma duracão próxima à 10 segundos.

|  Nome do arquivo | Tipo e Modelo | Dataset | Grupo | Durac. | Acc | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
|  **MLP_MFCC_model1_dataset_2_trim8** | **MLPmodel1** | **dataset2** | **trim8** | **8** | **1,00** | **0,94** |
|  **CNN_MFCC_model2_dataset_2_no_augment_10** | **CNNmodel2** | **dataset2** | **no_augment_10** | **10** | **1,00** | **0,88** |
|  **CNN_MFCC_model2_dataset_1_slide10** | **CNNmodel2** | **dataset1** | **slide10** | **10** | **1,00** | **0,80** |
|  CNN_MFCC_model2_dataset_1_trim5 | CNNmodel2 | dataset1 | trim5 | 5 | 0,99 | 0,77 |
|  CNN_MFCC_model2_dataset_2_no_augment_5 | CNNmodel2 | dataset2 | no_augment_5 | 5 | 0,97 | 0,77 |
|  CNN_MFCC_model2_dataset_2_trim8 | CNNmodel2 | dataset2 | trim8 | 8 | 0,97 | 0,73 |
|  MLP_MFCC_model1_dataset_2_slide10 | MLPmodel1 | dataset2 | slide10 | 10 | 1,00 | 0,73 |
|  CNN_MFCC_model2_dataset_1_trim8 | CNNmodel2 | dataset1 | trim8 | 8 | 0,97 | 0,66 |
|  MLP_MFCC_model1_dataset_1_trim8 | MLPmodel1 | dataset1 | trim8 | 8 | 1,00 | 0,66 |

## Discussão 

- Overffiting
- Topologia MLP
- Matrizes de confusão
- Datasets: qual utilizar?

### Overffiting

Ainda há bastante overffiting nos treinamentos. Esse sobreajuste pode ser facilmente observado nos gráficos de treinamento, porque as curvas de acurácia de treinamento rapidamente alcancam valores máximos. O gráfico de validacao tenta acompanhar, mas rapidamente atinge seu pico. O gráfico de perda ilustra o momento em que esse ponto de inflexão ocorre, momento no qual a perda de validacão comeca aumentar. 

A maioria dos modelos treinados obtiveram uma acurácia de treinamento igual a 1. Alguns modelos obtiveram um resultado satisfatório

### Mudanca nos modelos MLP

Os modelos de MLP sofreram muitas mudancas. Isso foi necessário porque o modelo era muito grande no experimento 2, mas observou-se que existe muito overffiting com este modelo. Nesse experimento, reduzimos o MLP para deixar o aprendizado mais adequado. No pŕoximo experimento, iremos reduzir ainda mais o modelo e ajustar seus hiperparâmetros para reduzir ainda mais o overffiting.

Apesar disso, as curvas dos treinamentos dos modelos MLP ainda apresenta uma característica estranha: o modelo capidamente converge e estabiliza.

### Topologia CNN

Parece que uma topologia com uma quantidade decrescente de filtros a cada camada produz um modelo interessante no escopo do problema.

### Matrizes de confusão

Observou-se nesse experimento uma característica interessante com respeito às matrizes de confusão. NO experimento anterior, as instâncias classificadas erradas tinham uma clara tendência em pertencer a uma determinada classe, na maioria das vezes, "HEalthy". Nesse experimento, os falso positivos e falso negativos ficaram "balanceados", isto é, o número de falso negativos e falso positivos ficaram muito próximos. Isso é interessante, porque indica que o modelo estava aprendendo características úteis para a classificacão.

### Datasets

Ainda não foi possível perceber muita diferença entre os datasets criados. A única conclusão no momento é que os dados com maiores durações produzem resultados melhores, mas não há muita evidência com relação ás técnicas de aumento de dados.

Pode ser interessante explorar mais um pouco alguns datasets nos próximos experimentos, como trim8, no_augment_10 e e slide10. 

A princípio o aumento de dados não atrapalhou no treinamento dos modelos, mas será necessário testar com dados não aumentados para garantir isso.

## Próximos passos

No próximo experimento iremos retreinar esses modelos e buscar otimizar os hiperparâmetros. Com relação aos dados, treinaremos com os conjuntos trim8, no_augment_10 e slide_10, e testaremos com os dados de teste de no_augment_10 e no_augment_8, para verificar se a técnica de aumento de dados está atrapalhando em algo.

## Conclusão

O experimento 3 mostrou que, mesmo com overfitting, os dados atuais podem ajudar para a construcão de um classificador para pneumonia. Uma pesquisa mais aprofundada nas topologias podem gerar modelos satisfatórios mesmo com essa quantidade limitada de dados.
