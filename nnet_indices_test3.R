# Morgan Bond, Lukas Naehrig, Lindsay Turner
# 8/30/2020
# Neural Network with Indices


## -- Required Packages: -- ##
library(caret)
library(doParallel)
library(e1071)
library(MLmetrics)
library(nnet)

setwd("C:/Users/lnaeh/Desktop/NOAA Research")

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
nsamples_class <- 1000
training_bc <- undersample_ds(dfAll, 'Classname', nsamples_class)
#table(training_bc$Classname)


## -- Split to training/test: -- ##
random_rows <- sort(sample(nrow(training_bc), nrow(training_bc)*.7))
training_data <- training_bc[random_rows,]
test_data  <- training_bc[-random_rows,]


## --Building the Neural Network Model: -- ##
start1 <- Sys.time()
model <- nnet(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2,
              data = training_data, size = 9, decay = 0.7, maxit = 1000)
end1 <- Sys.time()
end1 - start1

summary(model)
summary(model$residuals)

test_data$pred_nnet <- predict(model,test_data,type = 'class')
length(test_data$pred_nnet)
mtab <- table(as.factor(test_data$pred_nnet), test_data$Classname)
confusionMatrix(as.factor(test_data$pred_nnet), as.factor(test_data$Classname))



##########################################
### - With indices - ###

nre_fun <- function(x, y) {
  nre <- (y - x) / (y + x)
  return(nre)
}

#ndvi=nir2-red/nir2+red
ndvi_fun <- function(x, y) {
  ndvi <- (y - x) / (y + x)
  return(ndvi)
}

# ndwi (also worldview water index) = coastal-nir2/coastal+nir2
ndwi_fun <- function(x, y) {
  ndwi <- (y - x) / (y + x)
  return(ndwi)
}

# ccci=(nir2-rede)/(nir2-rede)/((nir2-red)/nir2+red)
ccci_fun <- function(x, y, z) {
  ccci <- ((y - x) / (x + y))/((y - z) / (y + z))
  return(ccci)
}

# BAI=blue-NIR/(blue+NIR)
bai_fun <- function(x, y) {
  bai <- (y - x) / (y + x)
  return(bai)
}
# REI=NIR2-Blue/NIR2+Blue*NIR2
rei_fun <- function(x, y) {
  rei<- (y - x) / (y + x*y)
  return(rei)
}

# Worldview built index (coastal-red edge)/(coastal+red edge)
wvbi_fun <- function(x, y) {
  wvbi <- (y - x) / (y + x)
  return(wvbi)
}

# NDSI normalized difference soil index (green-yellow)/(green+yellow)
ndsi_fun <- function(x, y) {
  ndsi <- (y - x) / (y + x)
  return(ndsi)
}

training_bc <- undersample_ds(dfAll, 'Classname', nsamples_class)
head(training_bc)

## -- Split to training/test: -- ##

training_bc$green.red <- as.vector(nre_fun(training_bc[3], training_bc[5]))
training_bc$blue.coastal <- as.vector(nre_fun(training_bc[2], training_bc[1]))
training_bc$NIR2.yellow <- as.vector(nre_fun(training_bc[8],training_bc[4]))
training_bc$NIR1.red <- as.vector(nre_fun(training_bc[7],training_bc[5]))
training_bc$rededge.yellow <- as.vector(nre_fun(training_bc[6],training_bc[4]))

training_bc$red.NIR2 <- as.vector(nre_fun(training_bc[5],training_bc[8]))
training_bc$rededge.NIR2 <- as.vector(nre_fun(training_bc[6],training_bc[8]))
training_bc$rededge.NIR1 <- as.vector(nre_fun(training_bc[6],training_bc[7]))
training_bc$green.NIR1 <- as.vector(nre_fun(training_bc[3],training_bc[7]))
training_bc$green.NIR2 <- as.vector(nre_fun(training_bc[3],training_bc[8]))

training_bc$rededge.green <- as.vector(nre_fun(training_bc[6],training_bc[3]))
training_bc$rededge.red <- as.vector(nre_fun(training_bc[6],training_bc[5]))
training_bc$yellow.NIR1 <- as.vector(nre_fun(training_bc[4],training_bc[7]))
training_bc$NIR2.blue <- as.vector(nre_fun(training_bc[8],training_bc[2]))
training_bc$blue.red <- as.vector(nre_fun(training_bc[2],training_bc[5]))

indices <- cbind(green.red, blue.coastal, NIR2.yellow, NIR1.red, rededge.yellow)
names(indices) <- c('green.red', 'blue.coastal', 'NIR2.yellow', 'NIR1.red', 'rededge.yellow')

# 10
indices <- cbind(green.red, blue.coastal, NIR2.yellow, NIR1.red, rededge.yellow,
                 red.NIR2, rededge.NIR2, rededge.NIR1, green.NIR1, green.NIR2)
names(indices) <- c('green.red', 'blue.coastal', 'NIR2.yellow', 'NIR1.red',
                    'rededge.yellow', 'red.NIR2', 'rededge.NIR2', 
                    'rededge.NIR1', 'green.NIR1', 'green.NIR2')

# 15
indices <- cbind(green.red, blue.coastal, NIR2.yellow, NIR1.red, rededge.yellow,
                 red.NIR2, rededge.NIR2, rededge.NIR1, green.NIR1, green.NIR2,
                 rededge.green, rededge.red, yellow.NIR1, NIR2.blue, blue.red)
names(indices) <- c('green.red', 'blue.coastal', 'NIR2.yellow', 'NIR1.red',
                    'rededge.yellow', 'red.NIR2', 'rededge.NIR2', 
                    'rededge.NIR1', 'green.NIR1', 'green.NIR2',
                    'rededge.green', 'rededge.red', 'yellow.NIR1',
                    'NIR2.blue','blue.red')

# Convert indices matrix to dataframe
indices_df <- as.data.frame(indices)
indices_df <- indices_df * 10000
indices_df$Classname <- training_bc$Classname
head(indices_df)

# Generate all indices using training data
indices <- matrix(data = NA, nrow = 11000, ncol = 64)
count <- 1
col_names <- character(64)
for (i in 1:8) {
  for (j in 1:8) {
    indices[, count] <- nre_fun(training_bc[,i], training_bc[,j]) * 10000
    col_names[count] <- paste(names(training_bc)[i], names(training_bc)[j], sep = ".")
    #col_names[count] <- paste("x",i,j,sep=".")
    count <- count + 1
  }
}

# Generate all indices without duplicates (1x8 and 8x1, or 1x1, 2x2, etc)
indices <- matrix(data = NA, nrow = 11000, ncol = 28)
count <- 1
col_names <- character(28)
for (i in 1:8) {
  for (j in i:8) {
    if(i != j) {
      indices[, count] <- nre_fun(training_bc[,i], training_bc[,j]) * 10000
      col_names[count] <- paste(names(training_bc)[i], names(training_bc)[j], sep = ".")
      #col_names[count] <- paste("x",i,j,sep=".")
      count <- count + 1
    }
  }
}

colnames(indices) <- col_names
indices_df <- as.data.frame(indices)
indices_df$Classname <- training_bc$Classname
head(indices_df)

### Training model
set.seed(1)
inTrain <- createDataPartition(y = indices_df$Classname,
                               ## the outcome data are needed
                               p = .90,
                               ## The percentage of data in the
                               ## training set
                               list = FALSE)
training <- indices_df[ inTrain,]
testing <- indices_df[-inTrain,]



##################################################################################
## -- Modeling -- ##
## Running models: Only indices:

start2 <- Sys.time()
modelIndices1 <- nnet(Classname ~ .,
                     data = training, size = 9, decay = 0.7, maxit = 1000)
end2 <- Sys.time()
end2 - start2
summary(modelIndices1)
summary(modelIndices1$residuals)

testing$pred_nnet <- predict(modelIndices1,testing, type = 'class')
length(testing$pred_nnet)
mtab <- table(as.factor(testing$pred_nnet), testing$Classname)
confusionMatrix(as.factor(testing$pred_nnet), as.factor(testing$Classname))


#################################################################################
# The part causing issues:
## Indices and raw values:

start3 <- Sys.time()
modelIndices2 <- nnet(Classname ~ .,
                     data = training_bc, size = 9, decay = 0.7, maxit = 1000)
end3 <- Sys.time()
end3 - start3

summary(modelIndices2)
summary(modelIndices2$residuals)

testing$pred_nnet <- predict(modelIndices2,testing, type = 'class')
length(testing$pred_nnet)
mtab <- table(as.factor(testing$pred_nnet), testing$Classname)
confusionMatrix(as.factor(testing$pred_nnet), as.factor(testing$Classname))
