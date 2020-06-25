## Lukas Naehrig
## 6/25/2020
## Binary Neural Network Model

setwd("C:/Users/lnaeh/Desktop/NOAA Research")
library(dplyr)
library(caret)
library(doParallel)
library(neuralnet)
library(nnet)
# 'vegan' package for scaling

## -- Loading data: -- ##
dfClass <- read.csv('geo_data.csv')
dfClass <-
  dfClass %>%
  rename('Classification' = 'Classname')

## -- Prognostics and sampling -- ##
set.seed(101)
size.sample <- 110
indeces <- dfClass[sample(1:nrow(dfClass), size.sample),] # get a training sample from dfClass
nnet_geoClass <- indeces
testSet <- dfClass[sample(1:nrow(dfClass), size.sample),]

# Binarize the categorical output
levels(dfClass$Classification)
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Barren')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Building')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'cloud')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'cultivated')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Deciduous Forest')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Dry sandbar')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Evergreen Forest')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Low streamside vegetation')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Road')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Shrubland')
nnet_geoClass <- cbind(nnet_geoClass, indeces$Classification == 'Water')

names(nnet_geoClass)[11] <- 'Barren'
names(nnet_geoClass)[12] <- 'Building'
names(nnet_geoClass)[13] <- 'cloud'
names(nnet_geoClass)[14] <- 'cultivated'
names(nnet_geoClass)[15] <- 'Deciduous_Forest'
names(nnet_geoClass)[16] <- 'Dry_sandbar'
names(nnet_geoClass)[17] <- 'Evergreen_Forest'
names(nnet_geoClass)[18] <- 'Low_streamside_vegetation'
names(nnet_geoClass)[19] <- 'Road'
names(nnet_geoClass)[20] <- 'Shrubland'
names(nnet_geoClass)[21] <- 'Water'

head(nnet_geoClass, 10) 

## -- Accuracy Funtion -- ##
weighted.acc <- function(predictions, actual) {
  freqs <- as.data.frame(table(actual))
  tmp <- t(mapply(function (p, a) { c(a, p==a) }, predictions, actual, USE.NAMES=FALSE)) # map over both together
  tab <- as.data.frame(table(tmp[,1], tmp[,2])[,2]) # gives rows of [F,T] counts, where each row is a state
  acc.pc <- tab[,1]/freqs[,2]
  return(sum(acc.pc)/length(acc.pc))
}


## -- Section 1: Neural Network -- ##
nn <- neuralnet(Barren + Building + cloud + cultivated + Deciduous_Forest +
                Dry_sandbar + Evergreen_Forest + Low_streamside_vegetation +
                Road + Shrubland + Water ~ 
                coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
                data = nnet_geoClass, hidden=c(3))
plot(nn) 

prediction <- predict(nn, newdata = testSet)
prediction





## -- Section 2: Scaled Neural Network -- ##

# # Scaling data:
# coastal_ <- scale(dfClass$coastal)
# blue_ <- scale(dfClass$blue)
# green_ <- scale(dfClass$green)
# yellow_ <- scale(dfClass$yellow)
# red_ <- scale(dfClass$red)
# rededge_ <- scale(dfClass$rededge)
# NIR1_ <- scale(dfClass$NIR1)
# NIR2_ <- scale(dfClass$NIR2)
# 
# newDF <- cbind(coastal_, blue_, green_, yellow_, red_, rededge_, NIR1_, NIR2_, dfClass$Classification)


