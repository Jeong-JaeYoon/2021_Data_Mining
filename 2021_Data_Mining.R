# Heart failure clinical records dataset

#library
library(spatstat);library(glmnet);library(ncpen)
library(ggplot2);library(ncvreg);library(dplyr)
library(caret);library(MLmetrics)
library(e1071);library(randomForest)
install.packages('randomForest')
#ready
rm(list=ls())
getwd()
setwd("C:\\Users\\jj950\\Desktop\\재윤\\건국대\\데이터 마이닝")
source("C:\\Users\\jj950\\Desktop\\재윤\\건국대\\데이터 마이닝\\data.mining.functions.2020.0602.R")     # 교수님 function불러오기
data = read.csv("archive\\heart_failure_clinical_records_dataset.csv")
data

#EDA 및 전처리

dim(data)
head(data)
summary(data)
str(data)

# 결측치 확인

colSums(is.na(data))

# train test data split

barplot(table(data$DEATH_EVENT), names.arg = c('survival', 'death'), main = "Death event")
# 2:1의 비율 확인, 데이터 분할 시 이 비율을 맞춰줘야함

set.seed(1234)

idx = createDataPartition(data$DEATH_EVENT, p = c(0.8, 0.2), list = FALSE)

train = data[idx,]
test = data[-idx,]

length(train$DEATH_EVENT)
length(test$DEATH_EVENT)
table(train$DEATH_EVENT)
table(test$DEATH_EVENT)

# train data 변수별 plot
par(mfrow=c(1,2))
plot(train$creatinine_phosphokinase, xlab = "index", ylab = "level", main = "Creatinine Phosphokinase")
plot(train$ejection_fraction, xlab = "index", ylab = "rate", main = "Ejection Fraction")
plot(train$serum_sodium, xlab = "index", ylab = "level", main = "Serum Sodium")
plot(train$platelets, xlab = "index", ylab = "level", main = "Platelets")
plot(train$serum_creatinine, xlab = "index", ylab = "rate", main = "Serum Creatinine")
plot(train$time, xlab = "index", ylab = "period", main = "Time")

# 이상치 발견
which(train$creatinine_phosphokinase > 5000)
which(train$serum_creatinine > 8)

# train 이상치 제거
train$creatinine_phosphokinase = ifelse(train$creatinine_phosphokinase > 5000, NA, train$creatinine_phosphokinase)
train$serum_creatinine = ifelse(train$serum_creatinine > 8, NA, train$serum_creatinine)

train = na.omit(train)
dim(train)

# valid, test 이상치 조건 동일히 적용
test$creatinine_phosphokinase = ifelse(test$creatinine_phosphokinase > 5000, NA, test$creatinine_phosphokinase)
test$serum_creatinine = ifelse(test$serum_creatinine > 8, NA, test$serum_creatinine)

test = na.omit(test)

# 변수별 정규성 확인
qqnorm(train[,1], main='age')
qqline(train[,1], col="red", lwd=2)
qqnorm(train[,3], main='creatinine phosphokinase')
qqline(train[,3], col="red", lwd=2)
qqnorm(train[,5], main='ejection_fraction')
qqline(train[,5], col="red", lwd=2)
qqnorm(train[,7], main='platelets')
qqline(train[,7], col="red", lwd=2)
qqnorm(train[,8], main='serum_creatinine')
qqline(train[,8], col="red", lwd=2)
qqnorm(train[,9], main='serum_sodium')
qqline(train[,9], col="red", lwd=2)
qqnorm(train[,12], main='time')
qqline(train[,12], col="red", lwd=2)

# creatinine phosphokinase, serum_creatinine, time은 정규성을 많이 벗어남, platelets는 애매함
par(mfrow=c(1,1))
ggplot(train, aes(x=time)) + geom_density() + labs(title = 'Follow up period', x = "Time", y = "density") + theme(plot.title = element_text(hjust = 0.5, face="bold", color = "blue"))
ggplot(train, aes(x=serum_creatinine)) + geom_density()+labs(title = 'Creatinine phosphokinase level', x = "level", y = "density") + theme(plot.title = element_text(hjust = 0.5, face="bold", color = "blue"))
ggplot(train, aes(x=platelets)) + geom_density()+labs(title = 'Platelets', x = "level", y = "density") + theme(plot.title = element_text(hjust = 0.5, face="bold", color = "blue"))
ggplot(train, aes(x=creatinine_phosphokinase)) + geom_density() + labs(title = 'Creatinine phosphokinase', x = "level", y = "density") + theme(plot.title = element_text(hjust = 0.5, face="bold", color = "blue"))

# time, platelets는 건들지 않기로 함
qqnorm(log(train$creatinine_phosphokinase))
qqline(log(train$creatinine_phosphokinase), col = "red", lwd = 2)
qqnorm(log(train$serum_creatinine))
qqline(log(train$serum_creatinine), col = "red", lwd = 2)

# creatineine phosphokinase, serum creatinine은 log 변환을 하기로 결정
train$creatinine_phosphokinase = log(train$creatinine_phosphokinase)
train$serum_creatinine = log(train$serum_creatinine)

test$creatinine_phosphokinase = log(test$creatinine_phosphokinase)
test$serum_creatinine = log(test$serum_creatinine)

# assessment F1 score를 사용.

# MLmetrics -> F1_score(y_pred, y_actual) 사용
# 위의 함수가 있으나 confusionMatrix를 사용해서 한 번 짜보기로 함
assess = function(y_pred, y){
  confusion = confusionMatrix(y_pred, y)
  precision = confusion$byClass['Pos Pred Value']
  recall = confusion$byClass['Sensitivity']
  f1 = (2*(precision*recall))/(precision + recall)
  return(f1)
}

# checking residuals
fit = glm(DEATH_EVENT~., data = train)
r.vec = fit$residuals
plot(r.vec)

# modeling
y.vec = as.vector(train[,13])
x.mat = as.matrix(train[,-13])

label = as.vector(test[,13])
feature = as.matrix(test[,-13])

m.vec = c("ridge","lasso","scad","mbridge")
m.vec = c(m.vec,paste("cv-",m.vec,sep=""))
b.mat = matrix(NA,nrow = 1+ncol(x.mat),ncol=length(m.vec))
colnames(b.mat)=m.vec
y_preds = NULL

#ridge
fit = glmnet(x=x.mat, y=y.vec, family = "binomial", alpha=0)
b.mat[,"ridge"] = coef(fit)[,length(fit$lambda)]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"ridge"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#lasso
fit = glmnet(x=x.mat, y=y.vec, family = "binomial", alpha=1)
b.mat[,"lasso"] = coef(fit)[,length(fit$lambda)]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"lasso"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#scad
library(ncpen)
fit = ncpen(y.vec,x.mat,family = "binomial",penalty = "scad")
b.mat[,"scad"] = coef(fit)[,length(fit$lambda)]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"scad"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#mbridge
fit = ncpen(y.vec,x.mat,family = "binomial",penalty = "mbridge")
b.mat[,"mbridge"] = coef(fit)[,length(fit$lambda)]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"mbridge"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#cross validation for ridge
cv.id= cv.index.fun(y.vec,k.val=10)
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=0,foldid=cv.id)
opt=which.min(cv.fit$cvm);opt
i = which(cv.fit$lambda == cv.fit$lambda.min);i
b.mat[,"cv-ridge"]=coef(cv.fit$glmnet.fit)[,opt]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"cv-ridge"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#cross validation for lasso
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=1,foldid=cv.id)
opt=which.min(cv.fit$cvm);opt
i = which(cv.fit$lambda == cv.fit$lambda.min);i
b.mat[,"cv-lasso"]=coef(cv.fit$glmnet.fit)[,opt]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"cv-lasso"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#cross validation for scad
cv.fit = cv.ncpen(y.vec,x.mat,family = "binomial",penalty = "scad",fold.id = cv.id)
opt=which.min(cv.fit$rmse);opt
b.mat[,"cv-scad"]=coef(cv.fit$ncpen.fit)[,opt]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"cv-scad"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#cross validation for mbridge
cv.fit = cv.ncpen(y.vec,x.mat,family = "binomial",penalty = "mbridge",fold.id = cv.id)
opt=which.min(cv.fit$rmse)
b.mat[,"cv-mbridge"]=coef(cv.fit$ncpen.fit)[,opt]
y_pred = cbind(1,as.matrix(feature))%*%as.matrix(b.mat[,"cv-mbridge"])
y_pred = ifelse(y_pred < 0, 0, 1)
y_preds = cbind(y_preds, y_pred)

#assessment
ass.mat = NULL
for (i in 1:ncol(y_preds)){
  F1 = F1_Score(y_preds[,i], label)
  ass.mat = cbind(ass.mat, F1)
}
colnames(ass.mat)=m.vec
which.max(ass.mat) #lasso

# final logistic model - Lasso

nx.mat = rbind(x.mat, feature)
ny.vec = c(y.vec,label)

fit = glmnet(x=x.mat, y=y.vec, family = "binomial", alpha=1)
b.mat = coef(fit)[,length(fit$lambda)]
y_pred = cbind(1,as.matrix(nx.mat))%*%as.matrix(b.mat)
y_pred = ifelse(y_pred < 0, 0, 1)
F1 = F1_Score(y_pred, ny.vec); F1

# SVM
# 데이터는 위의 데이터를 그대로 사용
svm.fit = svm(x = x.mat, y = y.vec, type = 'C-classification', kernel = 'sigmoid', epsilon = 0.1,cross = 10)
svm.pred = predict(svm.fit, newdata = feature)

summary(svm.fit)
str(svm.fit)
table(label,svm.pred)
svm.F1 = F1_Score(svm.pred, label)

# 전체 데이터에 대한 예측
svm.pred = predict(svm.fit, newdata = nx.mat)
svm.F1 = F1_Score(svm.pred, ny.vec); svm.F1

# SVM - gridsearch
cost.weight = c(0.01,0.1,1,10,100)
gamma.weight = c(0.1, 0.25, 0.5, 1)
tuning.results = tune(svm, train.x = x.mat, train.y = y.vec, kernel = "sigmoid", 
                      ranges = list(cost=cost.weight,gamma=gamma.weight))
print(tuning.results)

summary(tuning.results)
plot(tuning.results, cex.main=0.6, cex.lab=0.8, xaxs="i", yaxs="i")

svm.model.best = svm(x = x.mat, y = y.vec, type = 'C-classification', kernel = 'sigmoid', epsilon = 0.1, cost=0.1, gamma = 0.25)
svm.predictions.best = predict(svm.model.best, feature)

summary(svm.model.best)
str(svm.model.best)
table(label,svm.predictions.best)
svm.F1 = F1_Score(svm.predictions.best, label)

svm.best.pred = predict(svm.model.best, newdata = nx.mat)
svm.F1 = F1_Score(svm.best.pred, ny.vec); svm.F1

# random forest

rf = randomForest(x=x.mat, y=y.vec, ntree = 100)
summary(rf)
str(rf)
rf.pred = predict(rf,newdata = feature)
rf.pred = ifelse(rf.pred>0.5,1,0)
rf.F1 = F1_Score(rf.pred, label); rf.F1

rf.pred = predict(rf,newdata=nx.mat)
rf.pred = ifelse(rf.pred>0.5,1,0)
rf.F1 = F1_Score(rf.pred, ny.vec); rf.F1

ass.mat
b.mat
