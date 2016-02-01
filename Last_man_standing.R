### Analytics Vidhya Weekend Online Data Hack - Last Man Standing
### Date = 29/01/2016 - 31/01/2016
### Goal - We need to determine the outcome of the harvest season, 
### i.e. whether the crop would be healthy (alive), damaged by pesticides or damaged
### by other reasons.

## Load the require packages
require(caret) # for confusion matrix
require(randomForest) # for NA imputation
require(xgboost) # for model
require(h2o) # for anomaly detection
h2o.init(max_mem_size = "4g") # use all core and memory
h2o.removeAll() # remove prev cluster if there's any

# Set working directory
setwd("...\\Analytic_Vidhya\\The Toxic Pesticides\\Data")

# Load the dataset
train <- read.csv("Train.csv")
test <- read.csv("Test.csv")

# Fix NAs
train <- na.roughfix(train)
test <- na.roughfix(test)

# Convert training set into H2O format
train.hex <- as.h2o(train)

# Anomaly detection using H2O DL
dl_ae <- h2o.deeplearning(x = c(2:9),
                          training_frame = train.hex,
                          autoencoder = TRUE,
                          hidden = c(60, 60, 60),
                          epochs = 100)

model_anomaly <- h2o.anomaly(dl_ae, train.hex, per_feature = FALSE)
err <- as.data.frame(model_anomaly)
plot(sort(err$Reconstruction.MSE), main = "Reconstruction Error")
new_train <- train[err$Reconstruction.MSE < 1.2, ] # create a new training set til constant MSE

# Creating separate predictors
features <- names(new_train)[c(2:9)] 

# Correcting the target variable for xgboost model
new_train$Crop_Damage <- as.factor(new_train$Crop_Damage)
num.class <- length(levels(new_train$Crop_Damage))
y <- as.matrix(as.integer(new_train$Crop_Damage) - 1)

# Cross validation
xgb_cv <- xgb.cv(data = data.matrix(new_train[, features]),
                 label = y,
                 num_class = num.class,
                 objective = "multi:softprob",
                 nrounds = 100,
                 nfold = 10, prediction = TRUE)

# Checking minimum error from the rounds
min_err <- which.min(xgb_cv$dt[, test.merror.mean])
xgb_cv$dt[61, ] # 0.150209
# Checking training accuracy of the model 
pred.cv <- matrix(xgb_cv$pred, nrow = length(xgb_cv$pred)/num.class, ncol = num.class)
pred.cv <- max.col(pred.cv, "last")
confusionMatrix(factor(y+1), factor(pred.cv)) #  Accuracy : 0.8492

# Training the data with xgboost model
model <- xgboost(data = data.matrix(new_train[, features]),
                 label = y,
                 num_class = num.class,
                 objective = "multi:softprob", 
                 nrounds = min.err)

# Predicting on test set
pred <- predict(model, data.matrix(test[, features]))
pred <- matrix(pred, nrow = num.class, ncol = length(pred)/num.class)
pred <- t(pred)
pred <- max.col(pred, "last")

# Submission
submission <- data.frame(ID = test$ID, Crop_Damage = pred)
submission$Crop_Damage <- ifelse(submission$Crop_Damage == 1, 0, 
                                 ifelse(submission$Crop_Damage == 2, 1, 2))

write.csv(submission, "xgb_ae_model2.csv", row.names = FALSE)
# LB Score = 0.848340701774 (Among top 11%)
