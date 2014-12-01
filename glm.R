# Ejercicio de repaso - Regresion Logistica
rm(list = ls(all=TRUE))

# Saco notacion cientifica
options(scipen=666)

# setwd(path)
rm(list = ls(all=TRUE))
d = read.table("Adult.txt", sep="\t", header=TRUE)

### 1.1.1  
# Separe las poblaciones en entrenamiento y validación en forma aleatoria. 
# Indique que cantidad de casos quedaron para cada ambiente.

library(caret)
td <- sapply(d, class)

# Me fijo los valores unicos de cada variable
ln <- apply(d, 2, function(x) length(unique(x)))

# Si tiene más de 16 valores la convierto como factor, 
# después será convertida a categorica
fact = names(ln[ln < 17])

d <- d[order(d$Clase, decreasing=TRUE), ]

# Defino como factor a las nominales
for(j in fact) d[, j] <- as.factor(as.character(d[, j]))

d$Clase <- factor(ifelse(d$Clase == 1, 1, 0), levels=c(1, 0))

# d$Clase <- factor(ifelse(d$Clase == 1, 'ALTO', 'BAJO'), levels = c("ALTO", "BAJO"))
td <- sapply(d, class)

### Calculo cuantos NA hay por columna
na <- apply(d, 2, function(x) sum(is.na(x)))

print('variables con NAs')
na[na !=0]

# Los NAs estan en variables categoricas, agrego una categoria y listo
# Para eso defino a la funcion addNA
# d$WorkClass <- addNA(d$WorkClass)
# d$Occupation <- addNA(d$Occupation)

# Filtro registros con NAs
# Usar si no usas la funcion anterior
d <- d[complete.cases(d), ]

if(sum(is.na(d)) != 0)
  stop('Todavia hay NAs')

### Split en training y testing (sampleo estratificado por la clase) 
# Necesita al paquete caret
set.seed(123)
in_train <- createDataPartition(d$Clase, p=0.7, list=F)

train <- d[in_train, ]
test <- d[-in_train, ]

print(dim(train))
print(dim(test))

# 1.1.2  Formule el mejor modelo posible de regresión logística
library(leaps)

# Genero dos modelos el nulo y el completo
null <- glm(formula=Clase ~ 1, data=train, family='binomial')
full <- glm(formula=Clase ~ ., data=train, family='binomial')

# Foward
fwd <- step(null, scope=list(lower=null, upper=full), direction="forward", na.action='na.omit')
# Stepwise
step <- step(null, scope=list(lower=null, upper=full), direction="both", na.action='na.omit')
# Backward
back <- step(full, data=Housing, direction="backward", na.action='na.omit')

# Grabo lo anterior
# Se puede leer de nuevo usando load('glm.RData')

#load('glm.RData')

library(pROC)
library(caret)

plot_rocs <- function(train, test, model, titulo='ROC stepwise'){
  # Genera las predicciones de entrenamiento y testeo
  probs_train <- predict(model, newdata=train, type="response")
  probs_test <- predict(model, newdata=test, type="response")
  
  par(mfrow=c(1, 2))
  
  # Grafica la ROC de training en el cuadrante izquierdo
  roc_train <- plot.roc(Clase ~ probs_train, data=train,  
                        main=paste0(titulo, ' train'), percent=TRUE,  print.auc=TRUE,
                        thresholds="best", print.thres='best', col="#1c61b6")
  # Grafica la ROC de testing en el cuadrante derecho
  roc_test <- plot.roc(Clase ~ probs_test, data=test,  
                        main=paste0(titulo, ' test'), percent=TRUE,  print.auc=TRUE,
                       thresholds="best", print.thres='best', col="#008600")
  par(mfrow=c(1, 1))
  # Devuelve una lista con los siguientes elementos
  list(roc_train=roc_train, roc_test=roc_test, 
       probs_train=probs_train, probs_test=probs_test)
}

# 1.1.3 Calcular el AUC y las ROC en train y testing
roc_fwd <- plot_rocs(train, test, fwd, "ROC Foward")
roc_step <- plot_rocs(train, test, step, "ROC Stepwise")
roc_back <- plot_rocs(train, test, back, "ROC Backward")

# Stepwise y foward encuentran los mismos modelos (la formula es la misma)
identical(fwd$formula, step$formula)
# backward encuentra otra formula
identical(back$formula, step$formula)

# Genero una lista con todos los modelos
modelos <- list(fwd=fwd, step=step, back=back)

# Uso "[[" para extraer elementos por su nombre
# Formula de cada modelo
lapply(modelos, '[[', 'formula')

# Que variables fueron entrando y como vario el AIC
lapply(modelos, '[[', 'anova')

# 1.1.4  Selecciones el 10% de los trabajadores en el ambiente de validación de acuerdo a la siguiente lógica. 
# Entregue los resultados indicados:
# a)	Al azar e indique la cantidad de trabajadores que llegaron a superar los 50.000 dólares anuales.
# RTA: 230
# b)	Utilizando el modelo desarrollado en el punto 1.1.2 e 
# indique la cantidad de trabajadores que llegaron a superar los 50.000 dólares anuales.
# RTA: 798
# print(tbl)
# pred    1    0
#     1  798  124
#     0 1497 6796

set.seed(123)

in_validation <- createDataPartition(test$Clase, p=0.1, list=F)
validation <- test[in_validation, ]

# Porcentaje de clase = ALTO al azar en el 10%
clase_10pct <- sum(validation$Clase == 1)

contrasts(train$Clase)
probs <- 1 - predict(fwd, newdata=test, type="response")

# probs <- 1 - probs
quantile(probs, seq(0, 1, 0.1))
cuts <- quantile(probs, seq(0, 1, 0.1))
qt_90 <- quantile(probs, 0.9)

# probs > al percentil 90
# Es el 10% con mejor probabilidad 
sum(probs > qt_90[[1]])

pred <- factor(ifelse(probs > qt_90[[1]], 1, 0), levels=c(1, 0))
tbl <- table(pred, test$Clase)
tbl

# LIFT (a mano)
lift <- tbl[1,1] / clase_10pct
print(lift)


### ROC y Lift de RWD
library(ROCR)
# Curva ROC
pred <- prediction(probs, test$Clase)
roc_perf <- performance(pred, measure="tpr", x.measure="fpr")
plot(roc_perf, main="ROC Curve", colorize=T)

# Lift
perf_lift <- performance(pred, measure="lift", x.measure="rpp")
plot(perf_lift, main="lift curve", colorize=T)

# 1.1.5	Calcular y/o obtener los siguientes resultados: 
# a)	Indicar en cuanto sería el impacto en modificar una unidad de por lo menos una variable continua del modelo. 
# b)	Indicar si hay puntos incluyentes con COOK.
# c)	Indicar que método de selección de variables se utilizó y explicar su funcionamiento.
# d)	Mostrar el estadístico de Hosmer-Lemeshow en el último paso del modelo.


### a)
# Odds ratios
OR <- exp(data.frame(coef(fwd)))

# Variable Continua Age
# Ante un aumento en una unidad en la variable X, se espera que aumente sus odds en:
OR[row.names(OR) == 'Age', ]

par(mfrow=c(2,2))
plot(fwd)
par(mfrow=c(1,1))


### b)

# imI <- influence.measures(fwd)
cooks <- cooks.distance(fwd)
quantile(cooks, seq(0.9,1,0.01))

# Estas observaciones son puntos influyentes
drop_rows <- names(cooks[cooks > 0.00065237178])
train[drop_rows, ]


### c) Metodo de selección de variables: Stepwise.
# Forward selection, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.
# Backward elimination, which involves starting with all candidate variables, testing the deletion of each variable using a chosen model comparison criterion, deleting the variable (if any) that improves the model the most by being deleted, and repeating this process until no further improvement is possible.
# Bidirectional elimination, a combination of the above, testing at each step for variables to be included or excluded.


### d)
library('ResourceSelection')
hl <- hoslem.test(fwd$y, fitted(fwd), g=10)
cbind(hl$observed,hl$expected)
hl

save.image('glm.RData')


### AUC - GLMNET
glmnet::auc(as.numeric(as.character(test$Clase)), as.vector(probs))

y <- as.numeric(as.character(test$Clase))
prob <- as.vector(probs)

### AUC usando pROC
pROC::auc(y, prob)
plot.roc(y, prob)

# Calculo a mano del AUC (como lo hace glmnet, mucho más rápido)
rprob = rank(prob)
n1 = sum(y)
n0 = length(y) - n1
u = sum(rprob[y == 1]) - n1 * (n1 + 1)/2
u/(n1 * n0)
