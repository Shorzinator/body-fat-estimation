## Load data
setwd("D:/Wisconsin/628/project 2/")
file = read.csv("BodyFat.csv")
file = read.csv("preprocessed_data.csv",header = TRUE)
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
library(car)
library(dplyr)


# See relationship between bmi and circumferences
file3 = select(file,-c("BODYFAT","AGE","WEIGHT","HEIGHT"))
cor(file3) # BMI and all circumferences have rather high correlation, maybe delete some variables or apply PCA can be a solution
# Group Age according to the medical definition

file.copy1 = file
file$AGEGROUP = rep(0,nrow(file))
file$AGEGROUP[which(file$AGE<45)]=1
file$AGEGROUP[which(file$AGE>=45 & file$AGE<60)] = 2
file$AGEGROUP[which(file$AGE>=60)] = 3
file$AGEGROUP = as.factor(file$AGEGROUP)
file = dplyr::select(file,-c("AGE"))
unique(file$AGEGROUP)

library(dplyr)
# PCA towards circumferences to get common information, actually not use here
circum = dplyr::select(file,-c("AGEGROUP","BODYFAT","WEIGHT","HEIGHT","ADIPOSITY"))
m = prcomp(circum,center = TRUE, scale. = FALSE, retx = TRUE)
E = eigen(cov(circum))$vectors
lambda = eigen(cov(circum))$values
round(cumsum(lambda)/sum(lambda)*100, 1)
screeplot(m,type = "l")
cir1 = m$x[,1]
file.copy = file
file.copy$CIR1 = m$x[,1];file.copy$CIR2 = m$x[,2];file.copy$CIR3 = m$x[,3]
lm6 = lm(BODYFAT~.,file.copy1)
lm6 = lm(BODYFAT~.,file)
summary(lm6)
# lm6 basically conforms to linear regression assumptions and has low collinearity, a relatively good model
# but R2 is just 0.7.

# Robust regression
library(MASS)
rlm1 = rlm(BODYFAT~HEIGHT+WEIGHT+AGE+CIR1+CIR2+CIR3,file.copy)
summary(rlm1)
# All indicators perform still not so good

# Subset selection in lm
#library(haven)
#library(olsrr)
lmfit = lm(BODYFAT~.,data = file.copy1) 
lmfit = lm(BODYFAT~.,data = file) 
# gradual regression
lmfit.back = step(lmfit,direction = "both") # step is a function selecting model by AIC
summary(lmfit.back)
lm1 = lm(BODYFAT~WEIGHT+ABDOMEN+FOREARM+WRIST+AGEGROUP,file)
summary(lm1) # R2 0.7375, mean(vif)2.55. But both WEIGHT and ABDOMEN cannot be deleted
lm2 = lm(BODYFAT~AGE+WEIGHT+NECK+ABDOMEN+THIGH+FOREARM+WRIST,file.copy1)
summary(lm2) # R2 0.74, but high collinearity exist
# best subset based on BIC
library(leaps)
lmfit.cho = regsubsets(BODYFAT~.,file)
plot(lmfit.cho)
summary(lmfit.cho)
lm3.1 = lm(BODYFAT~WEIGHT+ABDOMEN,file.copy1)
summary(lm3.1) #0.7143, vif 4.65
lm3.2 = lm(BODYFAT~WEIGHT+ABDOMEN+FOREARM+WRIST,file.copy1)
summary(lm3.2) # 0.731, vif 3.93
lm3.3 = lm(BODYFAT~WEIGHT+ABDOMEN+WRIST,file.copy1)
summary(lm3.3) # 0.723, vif 4.33
lm3.4 = lm(BODYFAT~WEIGHT+NECK+ABDOMEN+FOREARM+WRIST,file.copy1)
summary(lm3.4) # 0.734, vif 4.13
lm3.5 = lm(BODYFAT~AGE+WEIGHT+ABDOMEN+THIGH+FOREARM+WRIST,file.copy1)
summary(lm3.5) # 0.737, vif 5.26
lm.cho2 = regsubsets(BODYFAT~.,data = file)
plot(lm.cho2)
lm4.1 = lm(BODYFAT~WEIGHT+ABDOMEN+FOREARM+WRIST,file)
summary(lm4.1) # 0.731, vif 3.93
lm4.2 = lm(BODYFAT~WEIGHT+ABDOMEN+FOREARM+WRIST+AGEGROUP,file) # I recommend
summary(lm4.2) # 0.738, vif 2.54
lm4.3 = lm(BODYFAT~WEIGHT+NECK+ABDOMEN+FOREARM+WRIST+AGEGROUP,file)
summary(lm4.3) # 0.740, vif 2.59
# One disappointing thing is high collinearity exists between weight and abdomen, but we cannot simply throw each one.
# Relatively, I think lm4.2 or lm3.4 can be good model
# CV, to validate result, method = "rlm" robust regression, "lm" linear regression
library(caret)
folds = trainControl(method = "repeatedcv",number = 10, repeats = 10) # 10 folds repeats 10 times
# lm4.2
## rlm
model.rlm1 = train(BODYFAT~WEIGHT+ABDOMEN+FOREARM+WRIST+AGEGROUP,data = file,method = "rlm",trControl = folds)
print(model.rlm1)
# robust bisquare with intercept, RMSE4.036, R2 0.732
## lm
model.lm1 = train(BODYFAT~.,data = file,method = "lm",trControl = folds)
print(model.lm1)
# RMSE 4.023, R2 0.732
# lm3.4
model.rlm2 = train(BODYFAT~WEIGHT+NECK+ABDOMEN+FOREARM+WRIST,data = file.copy1,method = "lm",trControl = folds)
print(model.rlm2)
# robust hampel, RMSE 4.047, R2 0.731
model.lm2 = train(BODYFAT~WEIGHT+NECK+ABDOMEN+FOREARM+WRIST,data = file.copy1,method = "lm",trControl = folds)
print(model.lm2)
# RMSE 4.036, R2 0.730
model.lm2 = train(BODYFAT~WEIGHT+NECK+ABDOMEN+FOREARM+WRIST,data = file.copy1,method = "lm",trControl = folds)
print(model.lm2)
lmz = lm(BODYFAT~WRIST+ABDOMEN+ADIPOSITY,file.copy1)

library(leaps)
lmfit.cho = regsubsets(BODYFAT~.,file)
plot(lmfit.cho)
lm5.1 = lm(BODYFAT~AGE+ABDOMEN+WRIST,file)
summary(lm5.1) # R2 0.729, vif 1.394
AIC(lm5.1);BIC(lm5.1) # AIC 187.66, BIC 205.27
lm5.2 = lm(BODYFAT~AGE+CHEST+ABDOMEN+WRIST+BMI,file)
summary(lm5.2) # R2 0.739, vif 5.358
AIC(lm5.2);BIC(lm5.2) # AIC 182.79, BIC 207.44
lm5.3 = lm(BODYFAT~AGE+NECK+ABDOMEN+WRIST,file)
summary(lm5.3) # R2 0.733, vif 2.194
AIC(lm5.3);BIC(lm5.3) # AIC 186.85, BIC 207.98
lm5.4 = lm(BODYFAT~AGE+CHEST+ABDOMEN+HIP+WRIST+BMI,file)
summary(lm5.4) # R2 0.742, vif 6.109
AIC(lm5.4);BIC(lm5.4) # AIC 181.84, BIC 210.008
lm5.5 = lm(BODYFAT~ABDOMEN+WRIST,file)
summary(lm5.5) # R2 0.717, vif 1.535
AIC(lm5.5);BIC(lm5.5) # AIC 197.33, BIC 211.42
lm5.6 = lm(BODYFAT~AGE+NECK+CHEST+ABDOMEN+HIP+WRIST+BMI,file)
summary(lm5.6) # R2 0.745, vif 5.88
AIC(lm5.6);BIC(lm5.6) # AIC 180.76, BIC, 212.46
lm6.1 = rlm(BODYFAT~AGE+ABDOMEN+WRIST,file)
AIC(lm6.1);BIC(lm6.1) # AIC 187.72, BIC 205.33, vif 1.39
lm6.2 = rlm(BODYFAT~AGE+CHEST+ABDOMEN+WRIST+BMI,file)
AIC(lm6.2);BIC(lm6.2) # AIC 182.87, BIC 207.53, vif 5.36
lm6.3 = rlm(BODYFAT~AGE+NECK+ABDOMEN+WRIST,file)
AIC(lm6.3);BIC(lm6.3) # AIC 186.92, BIC 208.05, vif 2.19
lm6.4 = rlm(BODYFAT~AGE+CHEST+ABDOMEN+HIP+WRIST+BMI,file)
AIC(lm6.4);BIC(lm6.4) # AIC 181.91, BIC 210.08, vif 6.11
lm6.5 = rlm(BODYFAT~ABDOMEN+WRIST,file)
AIC(lm6.5);BIC(lm6.5) # AIC 197.34, BIC 211.43, vif 1.53
lm6.6 = rlm(BODYFAT~AGE+NECK+CHEST+ABDOMEN+HIP+WRIST+BMI,file)
AIC(lm6.6);BIC(lm6.6) # AIC 180.84, BIC 212.54
