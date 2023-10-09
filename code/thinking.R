# Samples are somehow not enough
# visualization 
# select data based on real situation and outlier imputation/remove, use other indicators to verify
# data grouping, maybe testing differences between different groups
# apply existing tools to help variable choice
# PCA, robust regression

## Load data
setwd("D:/Wisconsin/628/project 2/")
file = read.csv("BodyFat.csv")
file = file[,-c(1,3)]
# transformation
file = file[which(file$BODYFAT!=0),] # remove bodyfat==0, which is impossible
sum(is.na(file))
# visualization
boxplot(file) # A few outliers exist

# height in inches, weight in lb(1 lb = 0.4536kg, 1 inch = 2.54cm)
bmi = file$WEIGHT*0.4536/(file$HEIGHT*2.54/100)^2
file$ADIPOSITY==bmi
head(bmi)
# roughly equal, adiposity contains information of height and weight

library(dplyr)
file2 = select(file,-c("WEIGHT","HEIGHT","ABDOMEN","HIP","CHEST",))
lm2 = lm(BODYFAT~.,file2)
summary(lm2)
vif(lm2)
mean(vif(lm2))

# See relationship between bmi and circumferences
file3 = select(file,-c("BODYFAT","AGE","WEIGHT","HEIGHT"))
lm3 = lm(ADIPOSITY~.,file3)
summary(lm3)
vif(lm3)
cor(file3) # BMI and all circumferences have rather high correlation, maybe delete some variables or apply PCA can be a solution

# model based on website, serious outliers exist, 39 R^2 is 0.6, which not high as expected.
file3 = file[-c(39),]
lm4 = lm(BODYFAT~AGE+NECK+ABDOMEN,file3)
summary(lm4)
vif(lm4)
plot(lm4)

# Group Age according to the medical definition
file.copy1 = file
file$AGEGROUP = rep(0,nrow(file))
file$AGEGROUP[which(file$AGE<45)]=1
file$AGEGROUP[which(file$AGE>=45 & file$AGE<60)] = 2
file$AGEGROUP[which(file$AGE>=60)] = 3
file$AGEGROUP = as.factor(file$AGEGROUP)
file = dplyr::select(file,-c("AGE"))
unique(file$AGEGROUP)
file4 = select(file,-c("WEIGHT","HEIGHT","ABDOMEN","HIP","CHEST","AGE"))
file4 = file4[-c(39),]
lm5 = lm(BODYFAT~AGEGROUP+WRIST+ADIPOSITY,file4)
summary(lm5)
plot(lm5)

# PCA towards circumferences to get common information
circum = dplyr::select(file,-c("AGE","AGEGROUP","BODYFAT","WEIGHT","HEIGHT","ADIPOSITY"))
m = prcomp(circum,center = TRUE, scale. = FALSE, retx = TRUE)
E = eigen(cov(circum))$vectors
lambda = eigen(cov(circum))$values
round(cumsum(lambda)/sum(lambda)*100, 1)
screeplot(m,type = "l")
cir1 = m$x[,1]
file.copy = file
file.copy$CIR1 = m$x[,1];file.copy$CIR2 = m$x[,2];file.copy$CIR3 = m$x[,3]
lm6 = lm(BODYFAT~HEIGHT+CIR1+CIR2+CIR3,file.copy)
summary(lm6)
# lm6 basically conforms to linear regression assumptions and has low collinearity, a relatively good model
# but R2 is just 0.7.

# Robust regression
library(MASS)
rlm1 = rlm(BODYFAT~HEIGHT+WEIGHT+AGEGROUP+CIR1+CIR2+CIR3,file.copy)
summary(rlm1)
# All indicators perform still not so good

# Subset selection in lm
#library(haven)
#library(olsrr)
lmfit = lm(BODYFAT~HEIGHT+WEIGHT+AGEGROUP+CIR1+CIR2+CIR3,data = file.copy)
lmfit = lm(BODYFAT~.,data = file.copy1)
# gradual regression
lmfit.back = step(lmfit,direction = "both")
summary(lmfit.back)
# The second lmfit reaches the highest R2 0.73
# best subset 
library(leaps)
lmfit.cho = regsubsets(BODYFAT~HEIGHT+WEIGHT+AGEGROUP+CIR1+CIR2+CIR3,file.copy) # weight+cir1,2,3
plot(lmfit.cho)
summary(lmfit.cho)
lm.cho2 = regsubsets(BODYFAT~.,data = file.copy1) # weight+abdomen+forearm+wrist
plot(lm.cho2)
summary(lm.cho2) # result same as gradual regression

# CV, to validate result, method = "rlm" robust regression, "lm" linear regression
library(caret)
folds = trainControl(method = "repeatedcv",number = 10, repeats = 10) # 10 folds repeats 10 times
model.rlm = train(BODYFAT~WEIGHT+AGEGROUP+HEIGHT+CIR1+CIR2+CIR3,data = file.copy,method = "rlm",trControl = folds)
# with PCA components
# The RMSE of psi.huber with intercept is 4.259, relatively smaller than other models and linear model.
model.rlm = train(BODYFAT~AGE+WEIGHT+NECK+ABDOMEN+HIP+THIGH+FOREARM+WRIST,data = file.copy1,method = "rlm",trControl = folds)
# model.rlm = train(BODYFAT~.,data = file,method = "rlm",trControl = folds)
# without PCA components
# RMSE of psi.hampel/huber method without intercept is 4.11 and R2 is 0.72
print(model.rlm)

# PCA actually increases RMSE and decreases R2, wondering what is happening. However. it decreases collinearity dramatically.
# As for me, I would choose WEIGHT+AGEGROUP+CIRCUMFERENCEs to build linear model. Both roubst regression and linear regression work.


library(DMwR2)
n = ncol(file.copy2)
file.copy2 = file.copy1
for(i in 3:n){
  quantile1 = quantile(file.copy2[,i],0.9)
  quantile2 = quantile(file.copy2[,i],0.1)
  file.copy2[(which(file.copy2[,i]>quantile1|file.copy2[,i]<quantile2)),i]=NA
}
dat1 <- knnImputation(file.copy2[,3:n],meth = 'weighAvg',scale = T)
dat1 = cbind(file.copy[,c(1,2)],dat1)
summary(lm(BODYFAT~.,dat1))

# Randomforest imputation
library(missForest)
dat2 <- missForest(file.copy2,ntree = 100)
dat2$ximp
summary(lm(BODYFAT~.,dat2$ximp))

library(mice)
imp=mice(file.copy2[3:n],method="norm.nob",m=1,maxit=1)
dat3=complete(imp)
dat3 = cbind(file.copy[,c(1,2)],dat3)
summary(lm(BODYFAT~.,dat3))

library(DMwR2)
iris2=file.copy2[3:n]
LOF=lofactor(iris2,k=5)
LOF
outlier.LOF=order(LOF,decreasing = T)[1:5] #输出LOF值5个最大的id作为异常
outlier.LOF
