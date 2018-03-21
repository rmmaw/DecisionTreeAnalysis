rm(list=ls())

#####################################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}   # Package function

needed = c("rpart","rattle","pROC","randomForest","ada","MASS") 
installIfAbsentAndLoad(needed)

library(ada) # ada() to do adaboost
library(e1071) # to use tune() method
library(randomForest) ### random forest from last class lab

churndata<-read.table("Assignment2TrainingData.csv",sep=",",header=T)
data <- na.omit(churndata)
cust_data <- data[,-1]
head(data)
nobs <- nrow(data)
set.seed(5082)
n <- nrow(cust_data)
cust_data_1 <- cust_data[,-1]

train <- sample(1:n, .8*n)
test <- (1:n)[-train]
trainset <- cust_data_1[train,]
testset <- cust_data_1[test,]

atrist = .26448
retain = 1600
lost = 11500
stay = .45
leave = .55


bestmodel <- ada(formula=Churn ~ .,data=trainset,iter=82,bag.frac=0.5,
                  control=rpart.control(maxdepth=2,cp=0.01,minsplit=45,xval=10))

test_vect <- c(rep(0,85))
cost_vect <- c(rep(0,85))

csv <- matrix( 0, nrow = 85, ncol = 6)
colnames(csv) <- c("Cutoff","False Postive Rate","False Negative Rate","True Postive Rate","True Negative Rate","Overall Error Rate")
test <- seq(0.01, 0.85, by = 0.01)

a <- 1
for (i in test){
  values  <- rep("No", nrow(testset))
  values[predict(boostmodel, newdata = testset, type = "prob")[,2] >= i] <- 'Yes'
  table1 <- table(testset$Churn, values,dnn=c("Actual", "Predicted"))
  if (is.na(table1[0][1])){
    No = c(0,0)
    table1 = cbind(table1,No)
    table1
  }
  cost <- (table1["No","Yes"]*retain)+(table1["Yes","No"]*lost)+(table1["Yes","Yes"]*leave*lost)+(table1["Yes","Yes"]*stay*retain)
  error <- error <- (table1["No","Yes"] + table1["Yes","No"])/sum(table1)
  test_vect[a] <- error
  cost_vect[a] <- cost
  csv[a,1] <- a/100
  csv[a,2] <- table1[1,1]/(table1[1,1]+table1[1,2])
  csv[a,3] <- table1[2,2]/(table1[2,1]+table1[2,2])
  csv[a,4] <- table1[2,1]/(table1[2,1]+table1[2,2])
  csv[a,5] <- table1[1,2]/(table1[1,1]+table1[1,2])
  csv[a,6] <- table1[1,1]/sum(table1)
  a <- a + 1
}
csv
write.csv(csv, file = "ConfusionMatrix.csv")
# Plot the error rate and cost associated with each cutoff Rate
plot(test,test_vect,type = "s", main = "Error rate for Cutoff Rate") 
points(which.min(test_vect)/100, test_vect[which.min(test_vect)], col = "red", cex = 2, pch = 16)
plot(test,cost_vect,type = "s", main = "Cost for Cutoff Rate") 
points(which.min(cost_vect)/100, cost_vect[which.min(cost_vect)], col = "red", cex = 2, pch = 16)
best <- which.min(cost_vect)

values[predict(boostmodel, newdata = testset, type = "prob")[,2] >= test[best]] <- 'Yes'
table1 <- table(testset$Churn, values,dnn=c("Actual", "Predicted"))
table1

sumcost <- ((table1["No","Yes"]*retain)+(table1["Yes","No"]*lost)+(table1["Yes","Yes"]*leave*lost)+(table1["Yes","Yes"]*stay*retain))
cost_per_cust <- sumcost/sum(table1)
cost_per_cust 
# My ada boost model is my best model created. As a group we decided that this will be our best model as well. 
# That is why we wrote to a csv with confusion matrices per cutoff rate. 
