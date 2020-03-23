# Trabalhando com classificação
# Iniciando a caminhada de machine learning

# Definindo o diretorio de trabalho
setwd("C:/MLR/CL")
getwd()

# Definição dopProblema de Negócio: Prevendo ocorrencia de cancer de mama
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Coletando os dados

# *************************************************************************************
# Os dados do cancer de mama incluem 569 observações de biopsias de cancer,
# cada um com 32 caracteristicas (variaveis). Uma caracteristica é um numero de 
# identificação (ID), outro é o diagnostico de cancer, e 30 são medidas laboratoriais
# numéricas. O diagnostico é codificado com "M" para indicar maligno ou "B"
# para indicar benigno.
# **************************************************************************************

dados <- read.csv("data/dataset.csv", stringsAsFactors = FALSE)

# Visualizando as variaveis
str(dados)
View(dados)

# Explorando os Dados

# Excluindo acoluna ID
# Independentemente do método de aprendizado de máquina, deve sempre excluidas 
# variáveis de id, caso contrario, isto pode levar a resultados errados porque o ID
# pode ser para unicamente "para prever cada exemplo". Por conseguinte, um modelo
# que inclui um identificador pode sofre sobreajuste (overfitting),
# e será muito dificil usá-lo para generalizar os dados
dados$id = NULL

# Ajustando o label da variável alvo
dados$diagnosis = sapply(dados$diagnosis, function(x) {ifelse(x =='M', 'Maligno', 'Benigno')})

# Tranformando a variavel algo como fator
table(dados$diagnosis)
dados$diagnosis <- factor(dados$diagnosis, levels = c("Benigno", "Maligno"), labels = c("Benigno", "Maligno"))
str(dados$diagnosis)

# Verificando a proporção
round(prop.table(table(dados$diagnosis)) * 100, digits = 1)

# Medidas de Tendência Central
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Criando uma função de normalização dos dados
normalizar <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

#Testando a função de normalização - os resultados devem ser identicos
normalizar(c(1, 2, 3, 4, 5))
normalizar(c(10, 20, 30, 40, 50))

# Normalizando os dados
dados_norm <- as.data.frame(lapply(dados[2:31], normalizar))
View(dados)
View(dados_norm)

# Treinando o modelo com o Knn

# Carregando pacote
library(class)

# Criando o conjunto de dados de treinoe de teste
# Esta não é unica forma de dividir o conjunto de dados - podemos usar outras formas
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Criando os labels (Rótulos) para os dados de treino e teste
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]
length(dados_treino_labels)
length(dados_teste_labels)

# Criando a versão1 domodelo 
modelo_knn_v1 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 21)

# o knn retorna um objeto do tipofator com as previsões para cada execução
summary(modelo_knn_v1)

# Avaliando e Interpretando o Modelo
# Carregando o gmodels
library(gmodels)

# Criando uma tabela cruzada dos dados previstos x dados atuais
# usando os dados de teste
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)

# Interpretando os resultados
# Taxa de acerto do modelo: 98% (acertou 98 em 100)

# Otimizando a performance do modelo
# sando a função scale() para  padronizar o z-score
?scale # (ajuda )
dados_z <- as.data.frame(scale(dados[-1]))

# verificando a transformação
summary(dados_z$area_mean)

# Criando novos datasets de treino e de teste
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]

# Labels
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]

# Reclassificando
modelo_knn_v2 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 17)

# Criando um Confusion Matrix
CrossTable(x = dados_teste_labels, y = modelo_knn_v2, prop.chisq = FALSE)

# A performance da segunda versão do modelo foi inferior a da primeira
# podemos mudar o valor do k

# Construindo um modelo com o algoritmo support vector machine (svm)

# Semente
set.seed(40)

# Prepara o dataset
dados <- read.csv("data/dataset.csv", stringsAsFactors = FALSE)
dados$id = NULL
dados[, 'index'] <- ifelse(runif(nrow(dados)) < 0.8,1,0)
View(dados)

# Dados de treino e teste
trainset <- dados[dados$index==1,]
testset <- dados[dados$index==0,]

# Obter o indice dos datasets
traincolNum <- grep('index', names(trainset))

# Remover os indices dos datasets
trainset <- trainset[,-traincolNum]
testset <- testset[,-traincolNum]

# Obter indice de coluna da variável target no conjunto de dados
typecolumn <- grep('diag', names(dados))

# Criando o modelo
# Ajustamos o kernel para radial, já que este conjunto de dados não tem um 
# plano linear que pode ser desenhado
library(e1071)
modelo_svm_v1 <- svm(diagnosis ~ .,
                     data = trainset,
                     type = 'C-classification',
                     kernel = 'radial')

# Previsões nos dados de treino
pred_train <- predict(modelo_svm_v1, trainset)

# percentual de previsões corretas com dataset de treino
mean(pred_train ==trainset$diagnosis)

# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, testset)

# percentual de previsões corretas com dataset de teste
mean(pred_test == testset$diagnosis)

# Confusion Matrix
table(pred_test, testset$diagnosis)














