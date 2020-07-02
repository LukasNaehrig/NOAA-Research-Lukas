# Lukas Naehrig
# 7/2/2020
# SVM & Boosted Modeling Attempt

## -- Required Packages: -- ##
setwd("C:/Users/lnaeh/Desktop/NOAA Research")
library(caret)
library(doParallel)
library(e1071)
library(MLmetrics)
library(kernlab)


## -- Data Frame import and cleaning: -- ##
dfAll <- read.csv('geo_data.csv')
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
nsamples_class <- 100
training_bc <- undersample_ds(dfAll, 'Classname', nsamples_class)
#table(training_bc$Classname)


## -- Split to training/test: -- ##
random_rows <- sort(sample(nrow(training_bc), nrow(training_bc)*.7))
training_data <- training_bc[random_rows,]
test_data  <- training_bc[-random_rows,]


## --Building the Neural Network Model: -- ##
# Linear
start1 <- Sys.time()
svmfit <- svm(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
              data = training_data, kernel = "linear", cost = 10, scale = F)
end1 <- Sys.time()
end1 - start1
print(svmfit)
summary(svmfit)
summary(svmfit$residuals)
test_data$pred_nnet <- predict(svmfit, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))

# Radial

start2 <- Sys.time()
svmfit2 <- svm(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
              data = training_data, kernel = "radial", cost = 10, scale = F)
end2 <- Sys.time()
end2 - start2
print(svmfit2)
summary(svmfit2)
summary(svmfit2$residuals)
test_data$pred_nnet <- predict(svmfit2, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))

# Polynomial

start3 <- Sys.time()
svmfit3 <- svm(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
              data = training_data, kernel = "polynomial", cost = 10, scale = F)
end3 <- Sys.time()
end3 - start3
print(svmfit3)
summary(svmfit3)
summary(svmfit3$residuals)
test_data$pred_nnet <- predict(svmfit3, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))


# Caret train function for:
# svmPoly
start4 <- Sys.time()
svmModel1 <- train(Classname ~ coastal + blue + green + 
                    yellow + red + rededge + NIR1 + NIR2,
                  data = training_data, 
                  method = 'svmPoly',
                  verbose = F)
end4 <- Sys.time()
end4 - start4
test_data$pred_nnet <- predict(svmModel1, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))

# svmRadial
start5 <- Sys.time()
svmModel2 <- train(Classname ~ coastal + blue + green + 
                    yellow + red + rededge + NIR1 + NIR2,
                  data = training_data, 
                  method = 'svmRadial',
                  verbose = F)
end5 <- Sys.time()
end5 - start5
test_data$pred_nnet <- predict(svmModel2, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))

# svmLinear2
start6 <- Sys.time()
svmModel3 <- train(Classname ~ coastal + blue + green + 
                    yellow + red + rededge + NIR1 + NIR2,
                  data = training_data, 
                  method = 'svmLinear2',
                  verbose = F)
end6 <- Sys.time()
end6 - start6
test_data$pred_nnet <- predict(svmModel3, test_data, type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))
