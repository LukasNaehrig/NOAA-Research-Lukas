## Lukas Naehrig
## 6/24/2020
## Caret, Neural Network Model
## Unscaled

#setwd("C:/Users/lnaeh/Desktop/NOAA Research")
library(dplyr)
library(caret)
library(doParallel)
library(neuralnet)

## -- Loading data: -- ##
dfClass <- read.csv('geo_data.csv')
dfClass <-
  dfClass %>%
  rename('Classification' = 'Classname')

## -- Prognostics and sampling prep -- ##
head(dfClass)
tail(dfClass)
n <- nrow(dfClass)

## -- Splitting data into train and test: -- ##
set.seed(111)
s <- sample(n, 0.9 * n, replace = F)
training <- dfClass[s,]
testing <- dfClass[-s,]

## -- Accuracy Funtion -- ##
weighted.acc <- function(predictions, actual) {
  freqs <- as.data.frame(table(actual))
  tmp <- t(mapply(function (p, a) { c(a, p==a) }, predictions, actual, USE.NAMES=FALSE)) # map over both together
  tab <- as.data.frame(table(tmp[,1], tmp[,2])[,2]) # gives rows of [F,T] counts, where each row is a state
  acc.pc <- tab[,1]/freqs[,2]
  return(sum(acc.pc)/length(acc.pc))
}


## -- Section 1: Neural Network Model -- ##
startTrain <- Sys.time()
nnetModel <- train(Classification ~ coastal + blue + green + 
                     yellow + red + rededge + NIR1 + NIR2, 
                   data = training, method = 'nnet', verbose = F)
endTrain <- Sys.time()
endTrain - startTrain
#smallNNet
accuracy1 <- predict(nnetModel, newdata = testing)
#accuracy1
weighted.acc(accuracy1, testing$Classification)


## Another way that yields the same accuracy:
# startTrain2 <- Sys.time()
# nn <- neuralnet(Classification ~ coastal + blue + green + 
#                   yellow + red + rededge + NIR1 + NIR2, 
#                 data = training, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
# endTrain2 <- Sys.time()
# endTrain2 - startTrain2
# accuracy2 <- predict(nn, newdata = testing)
# #accuracy2
# weighted.acc(accuracy2, smallTest$Classification)
# 
# 
# # Scaling data:
# temp1 <- subset(smallTrain, select = -c(Classification))
# temp2 <- subset(smallTrain, select = c(Classification))
# maxs <- apply(temp1, 2, max) 
# mins <- apply(temp1, 2, min)
# scaled <- as.data.frame(scale(temp1, center = mins, scale = maxs - mins))
# total <- rbind(scaled, temp2)
# train_ <- total[s,]
# test_ <- total[s1,]
# nn_ <- neuralnet(Classification ~ coastal + blue + green + 
#                    yellow + red + rededge + NIR1 + NIR2,
#                  data = train_, hidden = c(2,1), linear.output = F)
