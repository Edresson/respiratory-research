# Experimento 2

No experimento 2 foram analisados várias topologias e várias datasets. Neste experimento, foram criados dois conjuntos de datasets, separando aleatoriamente dois pacientes de cada instância (saudável e pneumonia) para o conjunto de teste. Durante a fase de treinamento, o conjunto de validacão foi gerado a partir do conjunto de treinamento.

Algumas conclusões interessantes foram obtidas, mas infelizmente houve muito overffiting.

|  Nome do arquivo | Tipo | Modelo | Tipo e Modelo | Dataset | Grupo | Durac. | Acc | Val acc | Test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  **CNN_MFCC_model2_dataset_2_no_augment_5** | **CNN** | **model2** | **CNNmodel2** | **dataset2** | **no_augment_5** | **5** | **1,00** | **0,67** | **0,81** |
|  **MLP_MFCC_model1_dataset_2_slide10** | **MLP** | **model1** | **MLPmodel1** | **dataset2** | **slide10** | **10** | **1,00** | **0,90** | **0,79** |
|  **CNN_MFCC_model2_dataset_1_slide10** | **CNN** | **model2** | **CNNmodel2** | **dataset1** | **slide10** | **10** | **1,00** | **1,00** | **0,77** |
|  **CNN_MFCC_model2_dataset_2_no_augment_10** | **CNN** | **model2** | **CNNmodel2** | **dataset2** | **no_augment_10** | **10** | **1,00** | **0,67** | **0,75** |
|  MLP_MFCC_model1_dataset_1_slide10 | MLP | model1 | MLPmodel1 | dataset1 | slide10 | 10 | 1,00 | 0,99 | 0,64 |
|  CNN_MFCC_model2_dataset_1_no_augment_10 | CNN | model2 | CNNmodel2 | dataset1 | no_augment_10 | 10 | 1,00 | 0,85 | 0,63 |
|  CNN_MFCC_model2_dataset_1_trim5 | CNN | model2 | CNNmodel2 | dataset1 | trim5 | 5 | 0,97 | 0,86 | 0,63 |
|  MLP_MFCC_model1_dataset_2_trim5 | MLP | model1 | MLPmodel1 | dataset2 | trim5 | 5 | 1,00 | 0,97 | 0,60 |
|  CNN_MFCC_model1_dataset_1_trim5 | CNN | model1 | CNNmodel1 | dataset1 | trim5 | 5 | 1,00 | 1,00 | 0,60 |
|  MLP_MFCC_model1_dataset_1_slide5 | MLP | model1 | MLPmodel1 | dataset1 | slide5 | 5 | 1,00 | 0,97 | 0,60 |
|  MLP_MFCC_model1_dataset_1_trim5 | MLP | model1 | MLPmodel1 | dataset1 | trim5 | 5 | 1,00 | 0,89 | 0,60 |
|  MLP_MFCC_model1_dataset_2_no_augment_5 | MLP | model1 | MLPmodel1 | dataset2 | no_augment_5 | 5 | 1,00 | 0,83 | 0,56 |
|  CNN_MFCC_model1_dataset_1_slide5 | CNN | model1 | CNNmodel1 | dataset1 | slide5 | 5 | 1,00 | 1,00 | 0,56 |
|  MLP_MFCC_model1_dataset_1_no_augment_5 | MLP | model1 | MLPmodel1 | dataset1 | no_augment_5 | 5 | 1,00 | 0,55 | 0,55 |
|  CNN_MFCC_model1_dataset_1_no_augment_5 | CNN | model1 | CNNmodel1 | dataset1 | no_augment_5 | 5 | 1,00 | 1,00 | 0,54 |
|  CNN_MFCC_model2_dataset_1_no_augment_5 | CNN | model2 | CNNmodel2 | dataset1 | no_augment_5 | 5 | 1,00 | 0,92 | 0,54 |
|  CNN_MFCC_model2_dataset_2_trim5 | CNN | model2 | CNNmodel2 | dataset2 | trim5 | 5 | 0,94 | 1,00 | 0,54 |
|  MLP_MFCC_model1_dataset_1_no_augment_10 | MLP | model1 | MLPmodel1 | dataset1 | no_augment_10 | 10 | 1,00 | 0,92 | 0,54 |
|  CNN_MFCC_model1_dataset_1_slide10 | CNN | model1 | CNNmodel1 | dataset1 | slide10 | 10 | 1,00 | 1,00 | 0,53 |
|  MLP_MFCC_model1_dataset_2_no_augment_10 | MLP | model1 | MLPmodel1 | dataset2 | no_augment_10 | 10 | 1,00 | 0,83 | 0,50 |
|  CNN_MFCC_model1_dataset_2_slide5 | CNN | model1 | CNNmodel1 | dataset2 | slide5 | 5 | 1,00 | 1,00 | 0,46 |
|  CNN_MFCC_model1_dataset_1_no_augment_10 | CNN | model1 | CNNmodel1 | dataset1 | no_augment_10 | 10 | 1,00 | 1,00 | 0,45 |
|  MLP_MFCC_model1_dataset_2_slide5 | MLP | model1 | MLPmodel1 | dataset2 | slide5 | 5 | 1,00 | 0,89 | 0,45 |
|  CNN_MFCC_model1_dataset_2_slide10 | CNN | model1 | CNNmodel1 | dataset2 | slide10 | 10 | 1,00 | 1,00 | 0,44 |
|  CNN_MFCC_model2_dataset_1_slide5 | CNN | model2 | CNNmodel2 | dataset1 | slide5 | 5 | 1,00 | 1,00 | 0,42 |
|  CNN_MFCC_model2_dataset_2_slide10 | CNN | model2 | CNNmodel2 | dataset2 | slide10 | 10 | 1,00 | 1,00 | 0,41 |
|  CNN_MFCC_model2_dataset_2_slide5 | CNN | model2 | CNNmodel2 | dataset2 | slide5 | 5 | 1,00 | 1,00 | 0,38 |
|  CNN_MFCC_model1_dataset_2_trim5 | CNN | model1 | CNNmodel1 | dataset2 | trim5 | 5 | 1,00 | 1,00 | 0,37 |
|  CNN_MFCC_model1_dataset_2_no_augment_10 | CNN | model1 | CNNmodel1 | dataset2 | no_augment_10 | 10 | 1,00 | 1,00 | 0,25 |
|  CNN_MFCC_model1_dataset_2_no_augment_5 | CNN | model1 | CNNmodel1 | dataset2 | no_augment_5 | 5 | 1,00 | 1,00 | 0,18 |

## Discussão

- Overfitting
- Dataset pequeno
- Conjunto de dados (slide5, trim5...)

### Overfitting

O overfitting se mostrou bastante presente nos experimentos realizados, principalmente nos modelos MLP. Verificou-se que em vários experimentos, o desempeho de validacão ficou próximo do desempenho de treinamento, e em alguns casos chegou a mais de 90% de acurácia. Em geral, o desempenho do teste se mostrou ligeiramente inferior, em torno de 60% e 70% de acurácia. 

O modelo MLP evidenciou mais o overfitting. É fácil ver que o modelo satura nas primeiras épocas, impossibilitando o aprendizado. Isso também evidencia que a quantidade de dados para treinamento é insuficiente para esse tipo de dado. DIminuir a rede ou adicionar dropout pouco ajuda: o modelo passa a aprender pouco, e aumentar a rede produz mais overfitting.

A maioria dos treinamentos teve resultado pouco satisfatório: em torno de 60%, o que é bem ruim já que se trata de uma class. binária. O resultado da validac. em alguns casos chegou a 100%, o que sugere um forte overffiting.

### Dataset Pequeno

O dataset se mostrou pequeno, porque os modelos saturam muito rápido.

### Dados desbalanceados

Os resultados dos experimentos mostraram que será fundamental realizar o balanceamento dos dados. Perceba que nas matrizes de confusão os erros evidenciam isso.

### Datasets

O dataset 2 teve um desempenho ligeiramente pior, mas provavelmente porque é o dataset mais desbalanceado.

### Conjunto de dados

10 segundos parece ser melhor, tanto slide10 quanto no_augment_10 tiveram resultados melhores. Os experimentos sugerem que uma duracão maior é mais adequada, entretanto, não é possível faz trimming de 10 segundos. Por isso, pode ser interessante trabalhar com uma duracão menor de trimming, como 8 segundos. 

## Próximos passos: escolher estratégia e maximizar treinamento

- Selecionar bons kernels e treinar novamente, ajustando hiperparâmetros e utilizando o conjunto de teste como validacao, para maximizar o conjunto de treinamento. 
- Será necessário trabalhar o _overfitting_.

## Anotacoes

- Para o ModelCheckpoint: Monitor val loss é melhor que val acc;
- Será necessário balancear os dados, pois houve bastante tendência em classificar os áudios como 'Healthy', classe com mais instâncias nos conjuntos de treinamento.
- Será necessário utilizar bastante regularizacao para mitigar o overfitting.
- O modelo 2 parece aprender melhor: trabalhar com ele e diminuir modelo para forcar a rede a aprender o que é necessário.

## Conclusão

- Mitigar overffiting com bastante regularizacao;
- Maximizar conjunto de treinamento;
- Focar em uma topologia mais promissora (provavelmenta CNN1);
- Balancear dados duplicando instâncias;
- Analisar qual estratégia para gerar datasets é a melhor;
- Salvar imagens e acurácias em CSV;
- CNN model 2 e MLP model 1 são promissores;

## Trabalhar com:

- Datasets: dataset 1 e 2 balanceados
- Grupos: Trim 8, Slide 10, No Augment 10
- Topologias: MLP1 e CNN2
- Conjunto de validacão igual ao conjunto de teste para maximizar o treinamento
- Técnicas de regularizacão e diminuir a rede para combater o overffiting
