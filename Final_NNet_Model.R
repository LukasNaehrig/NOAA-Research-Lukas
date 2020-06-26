# Morgan Bond, Lukas Naehrig
# 6/26/2020
# Neural Network with Grid Tuning
# Runtime: ~ 3 days


## -- Required Packages: -- ##
library(caret)
library(doParallel)
library(e1071)
library(MLmetrics)
library(nnet)


## -- Data Frame import and cleaning: -- ##
dfAll <- read.csv('geo_data.csv')           #### NOTE: THIS WILL NEED TO BE CHANGED
table(dfAll$Classname)
dfAll$Classname <- as.character(dfAll$Classname)
dfAll$Classname[dfAll$Classname == 'Low streamside vegetation'] <- 'Low.streamside.vegetation'
dfAll$Classname[dfAll$Classname == 'Deciduous Forest'] <- 'Deciduous.Forest'
dfAll$Classname[dfAll$Classname == 'Dry sandbar'] <- 'Dry.sandbar'
dfAll$Classname[dfAll$Classname == 'Evergreen Forest'] <- 'Evergreen.Forest'
dfAll$Classname <- as.factor(dfAll$Classname)
#levels(dfAll$Classname)

## -- Undersampling: -- ##
set.seed(12)
undersample_ds <- function(x, classCol, nsamples_class) {
  for (i in 1:length(unique(x[, classCol]))) {
    class.i <- unique(x[, classCol])[i]
    if ((sum(x[, classCol] == class.i) - nsamples_class) != 0) {
      x <- x[-sample(which(x[, classCol] == class.i),
                     sum(x[, classCol] == class.i) - nsamples_class), ]
    }
  }
  return(x)
}
nsamples_class <- 300000              ######## Maybe more or less than 300000
training_bc <- undersample_ds(dfAll, 'Classname', nsamples_class)
#table(training_bc$Classname)


## -- Split to training/test: -- ##
random_rows <- sort(sample(nrow(training_bc), nrow(training_bc)*.7))
training_data <- training_bc[random_rows,]
test_data  <- training_bc[-random_rows,]


## --Building the Neural Network Model: -- ##
model <- nnet(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
              data = training_data, size = 9, decay = 0.7, maxit = 1000)

summary(model)
summary(model$residuals)

test_data$pred_nnet <- predict(model,test_data,type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))

## -- Grid Tuning to Optimize Model Accuracy -- #
#require(caret)
#cl <- makeCluster(detectCores())
#registerDoParallel(cl)
#nnTrControl <- trainControl(classProbs = T, summaryFunction = multiClassSummary, allowParallel = T)
#my.grid <- expand.grid(.decay = c(0.7, 0.8, 0.9), .size = c(8, 9))
#mynnetfit <- train(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
#                   data = training_data, method = 'nnet', maxit = 1000, tuneGrid = my.grid,
#                   trace = F, trControl = nnTrControl)
#stopCluster(cl)
#plot(mynnetfit)






