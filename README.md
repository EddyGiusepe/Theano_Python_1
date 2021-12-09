# Data Scientist Jr.: Dr.Eddy Giusepe Chirinos Isidro

## Lesson04: Curso intensivo em percepções multicamadas
As redes neurais artificiais são uma área de estudo fascinante, embora possam ser
intimidantes no início. O campo das redes neurais artificiais costuma ser chamado apenas de redes 
neurais ou Perceptrons multicamadas, talvez devido ao tipo mais útil de rede neural. O bloco de
construção das redes neurais são os ``neurônios artificiais``. Estas são unidades computacionais
simples que têm sinais de entrada ponderados e produzem um sinal de saída usando uma função
de ativação. Os neurônios são organizados em redes de neurônios. Uma linha de neurônios é
chamada de camada e uma rede pode ter várias camadas. A arquitetura dos neurônios na rede costuma
ser chamada de topologia de rede. Depois de configurada, a rede neural precisa ser ``treinada`` em
seu conjunto de dados. O algoritmo de treinamento clássico e ainda preferido para redes neurais
é chamado de ``descida gradiente estocástica`` (stochastic gradient descent). 

Seu objetivo nesta lição é se familiarizar com a terminologia da rede neural. Aprofunde-se um pouco mais em termos
como <font color="yellow">neurônio</font>, <font color="yellow">pesos</font>, <font color="yellow">função
de ativação</font>, <font color="yellow">taxa de aprendizagem</font> e muito mais.

## Lesson05: Minha primeira rede neural em Keras
Keras permite que você desenvolva e avalie modelos de aprendizado profundo em poucas linhas
de código. Nesta lição , seu objetivo é desenvolver sua <font color="yellow">primeira rede neural</font>
usando a biblioteca ``Keras``. Use um conjunto de dados de classificação binária padrão 
(duas classes) do <font color="red">UCI Machine Learning Repository</font>, como o [aparecimento
de diabetes nos índios Pima](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv?__s=b0ntu45iywktlbv07dti&utm_source=drip&utm_medium=email&utm_campaign=DLWP+Mini-Course&utm_content=Lesson+05%3A+Your+First+Neural+Network+in+Keras) ou os conjuntos de dados da [ionosfera](https://archive.ics.uci.edu/ml/datasets/Ionosphere?__s=b0ntu45iywktlbv07dti&utm_source=drip&utm_medium=email&utm_campaign=DLWP+Mini-Course&utm_content=Lesson+05%3A+Your+First+Neural+Network+in+Keras).

Reúna o código para alcançar o seguinte:

1. Carregue seu conjunto de dados usando NumPy ou Pandas.
2. Defina seu modelo de rede neural e compile-o.
3. Ajuste seu modelo ao conjunto de dados.
4. Faça uma estimativa do desempenho do seu modelo com base em dados não vistos.


Para lhe dar uma grande vantagem, abaixo está um exemplo de trabalho completo que você pode 
usar como ponto de partida. Ele assume que você baixou o conjunto de dados Pima Indians para seu
diretório de trabalho atual com o nome de arquivo <font color="orange">pima-indians-diabetes.csv</font>.