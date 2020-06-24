## Lukas Naehrig
## 6/22/2020
## Caret, Neural Network, Support Vector Machine,
## ADA Boost

setwd("C:/Users/lnaeh/Desktop/NOAA Research")
library(dplyr)
library(caret)
library(doParallel)
library(neuralnet)
library(fastAdaboost)
library(e1071)
library(LiblineaR)

## -- Loading data: -- ##
dfClass <- read.csv('geo_data.csv')
dfClass <-
  dfClass %>%
  rename('Classification' = 'Classname')

## -- Prognostics and sampling -- ##
head(dfClass)
tail(dfClass)
n <- nrow(dfClass)
set.seed(123)
# scaleddata <- scale(dfClass)            ### SCALING?
s <- sample(n, 11000, replace = F)
smallTrain <- dfClass[s,]
s1 <- sample(n, 11000, replace = F)
smallTest <- dfClass[s1,]

## -- Accuracy Funtion -- ##
weighted.acc <- function(predictions, actual) {
  freqs <- as.data.frame(table(actual))
  tmp <- t(mapply(function (p, a) { c(a, p==a) }, predictions, actual, USE.NAMES=FALSE)) # map over both together
  tab <- as.data.frame(table(tmp[,1], tmp[,2])[,2]) # gives rows of [F,T] counts, where each row is a state
  acc.pc <- tab[,1]/freqs[,2]
  return(sum(acc.pc)/length(acc.pc))
}


## -- Splitting data into train and test: -- ##
# training <- dfClass[s,]
# testing  <- dfClass[-s,]

## -- Section 1: Neural Network -- ##

# set.seed(825)
smallNNet <- train(Classification ~ coastal + blue + green + 
                   yellow + red + rededge + NIR1 + NIR2, 
                   data = smallTrain, method = 'nnet', verbose = F)
smallNNet
accuracy1 <- predict(smallNNet, newdata = smallTest)
accuracy1
weighted.acc(accuracy1, smallTest$Classification)

# Another way that yields the same accuracy:

nn <- neuralnet(Classification ~ coastal + blue + green + 
                yellow + red + rededge + NIR1 + NIR2, 
                data = smallTrain, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
accuracy2 <- predict(smallNNet, newdata = smallTest)
accuracy2
weighted.acc(accuracy2, smallTest$Classification)


# Scaling data:
temp1 <- subset(smallTrain, select = -c(Classification))
temp2 <- subset(smallTrain, select = c(Classification))
maxs <- apply(temp1, 2, max) 
mins <- apply(temp1, 2, min)
scaled <- as.data.frame(scale(temp1, center = mins, scale = maxs - mins))
total <- rbind(scaled, temp2)
train_ <- total[s,]
test_ <- total[s1,]
nn_ <- neuralnet(Classification ~ coastal + blue + green + 
                 yellow + red + rededge + NIR1 + NIR2,
                 data = train_, hidden = c(2,1), linear.output = F)

## -- Section 2: Support Vector Machine -- ##
#start1 <- Sys.time()
#svmModel <- train(Classification ~ coastal + blue + green + 
#                  yellow + red + rededge + NIR1 + NIR2,
#                  data = smallTrain, 
#                  method = 'svmLinearWeights',
#                  verbose = F)
#end1 <- Sys.time()
#time1 <- end1 - start1
##svmModel
#accuracy3 <- predict(svmModel, newdata = smallTest)
#weighted.acc(accuracy3, smallTest$Classification)
#

start4 <- Sys.time()
svmModel <- train(Classification ~ coastal + blue + green + 
                  yellow + red + rededge + NIR1 + NIR2,
                  data = smallTrain, 
                  method = 'svmLinear3',
                  verbose = F)
end4 <- Sys.time()
time4 <- end4 - start4
time4
accuracy2 <- predict(smallNNet, newdata = smallTest)
accuracy2
weighted.acc(accuracy2, smallTest$Classification)

### DOESN'T WORK... WHY? ###
## -- Section 3: ADA Boost -- ##
start2 <- Sys.time()
adaModel <- train(Classification ~ coastal + blue + green + 
                  yellow + red + rededge + NIR1 + NIR2,
                  data = smallTrain, 
                  method = 'adaboost',
                  verbose = F)
end2 <- Sys.time()
time2 <- end2 - start2
#adaModel
accuracy4 <- predict(adaModel, newdata = smallTest)
weighted.acc(accuracy4, smallTest$Classification)


time1
weighted.acc(accuracy3, smallTest$Classification)
time2
weighted.acc(accuracy4, smallTest$Classification)



####### Different attempts:


#dat = data.frame(Classification, y = as.factor(y))
svmfit = svm(Classification ~ ., data = training, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)


# Things to do:
# more online research on nnet
# potentially expand to other models (adaboost, svm)
# add indeces maybe?
# run in lab for larger dataset.