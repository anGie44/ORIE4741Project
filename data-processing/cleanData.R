library(boot)
set.seed(1)
setwd("/Users/apinilla/GitProjects/anGie44/ORIE4741Project")
data_train<-read.csv("HCMST_train.csv", header=TRUE)
data_test<-read.csv("HCMST_test.csv", header=TRUE)


glm.train_fit<-glm(data_train$Q34~data_train$HHINC, data=data_train, na.action = na.omit)
x_train_pred<-predict(glm.train_fit, type="response")
x_test_pred<-predict(glm.train_fit, newdata = data_test, type="response")


library(e1071)
model<-svm(data_train$Q34~data_train$HHINC, data = data_train, probability=TRUE)
res<-predict(model, newdata = data_test, probability = TRUE)

table(pred = res, true = data_test$Q34)

for (i in 1:ncol(data_train)) {
  plot(data_train[,i], data_train$Q34)
}