# Load necessary libraries
library(glmnet)
library(ROCR)
library(dplyr)

# 1. Download and read the data
url <- "https://liangfgithub.github.io/Data/Caravan.csv"
download.file(url, destfile = "Caravan.csv")
caravan <- read.csv("Caravan.csv", stringsAsFactors = FALSE)

# Convert 'Purchase' to a binary variable
caravan$Purchase <- ifelse(caravan$Purchase == "Yes", 1, 0)

# 2. Split the data into test and training sets
test_indices <- 1:1000
test_data <- caravan[test_indices, ]
train_data <- caravan[-test_indices, ]

# Separate predictors and response
train_X <- train_data[, -which(names(train_data) == "Purchase")]
train_Y <- train_data$Purchase

test_X <- test_data[, -which(names(test_data) == "Purchase")]
test_Y <- test_data$Purchase

# Function to calculate misclassifications and AUC
calculate_metrics <- function(pred_probs, true_labels, cutoff = 0.25) {
  predicted_labels <- ifelse(pred_probs >= cutoff, 1, 0)
  
  # Misclassifications
  no_indices <- which(true_labels == 0)
  yes_indices <- which(true_labels == 1)
  
  misclass_no <- sum(predicted_labels[no_indices] == 1)
  misclass_yes <- sum(predicted_labels[yes_indices] == 0)
  
  # AUC
  pred <- prediction(pred_probs, true_labels)
  perf <- performance(pred, "auc")
  auc <- as.numeric(perf@y.values)
  
  return(list(misclass_no = misclass_no, 
              misclass_yes = misclass_yes, 
              auc = round(auc, 3)))
}

### Method 1: Logistic Regression with All 85 Predictors ###

# Fit logistic regression model
full_model <- glm(Purchase ~ ., data = train_data, family = binomial)

# Predict probabilities on test data
full_pred_probs <- predict(full_model, newdata = test_data, type = "response")

# Calculate metrics
metrics_full <- calculate_metrics(full_pred_probs, test_Y)

# Assign values
a1 <- metrics_full$misclass_no
b1 <- metrics_full$misclass_yes
c1 <- metrics_full$auc

# Print results for Method 1
cat("Method 1: Logistic Regression with All Predictors\n")
cat("Misclassified 'No':", a1, "\n")
cat("Misclassified 'Yes':", b1, "\n")
cat("AUC:", c1, "\n\n")

### Method 2: Forward Variable Selection using AIC ###

# Start with null model
null_model <- glm(Purchase ~ 1, data = train_data, family = binomial)

# Perform forward selection
forward_aic <- step(null_model, 
                    scope = list(lower = null_model, upper = full_model), 
                    direction = "forward", 
                    trace = FALSE)

# Number of non-intercept predictors
d2 <- length(coef(forward_aic)) - 1

# Predict probabilities on test data
forward_aic_probs <- predict(forward_aic, newdata = test_data, type = "response")

# Calculate metrics
metrics_forward_aic <- calculate_metrics(forward_aic_probs, test_Y)

# Assign values
a2 <- metrics_forward_aic$misclass_no
b2 <- metrics_forward_aic$misclass_yes
c2 <- metrics_forward_aic$auc

# Print results for Method 2
cat("Method 2: Forward Selection using AIC\n")
cat("Number of predictors (d2):", d2, "\n")
cat("Misclassified 'No':", a2, "\n")
cat("Misclassified 'Yes':", b2, "\n")
cat("AUC:", c2, "\n\n")

### Method 3: Forward Variable Selection using BIC ###

# Perform forward selection with BIC (k = log(n))
forward_bic <- step(null_model, 
                    scope = list(lower = null_model, upper = full_model), 
                    direction = "forward", 
                    k = log(nrow(train_data)),
                    trace = FALSE)

# Number of non-intercept predictors
d3 <- length(coef(forward_bic)) - 1

# Predict probabilities on test data
forward_bic_probs <- predict(forward_bic, newdata = test_data, type = "response")

# Calculate metrics
metrics_forward_bic <- calculate_metrics(forward_bic_probs, test_Y)

# Assign values
a3 <- metrics_forward_bic$misclass_no
b3 <- metrics_forward_bic$misclass_yes
c3 <- metrics_forward_bic$auc

# Print results for Method 3
cat("Method 3: Forward Selection using BIC\n")
cat("Number of predictors (d3):", d3, "\n")
cat("Misclassified 'No':", a3, "\n")
cat("Misclassified 'Yes':", b3, "\n")
cat("AUC:", c3, "\n\n")

### Method 4: L1 Penalty (Lasso) using glmnet ###

# Prepare data for glmnet
# glmnet requires matrices
train_matrix <- model.matrix(Purchase ~ ., data = train_data)
test_matrix <- model.matrix(Purchase ~ ., data = test_data)

# Fit lasso model with lambda = 0.004
lasso_model <- glmnet(train_matrix, train_Y, lambda = 0.004, standardize = TRUE, intercept = TRUE, family = "binomial")

# Extract coefficients
lasso_coefs <- coef(lasso_model, s=0.004)
# Number of non-intercept predictors
d4 <- sum(lasso_coefs[-1] != 0)

# Predict probabilities on test data
lasso_pred_probs <- predict(lasso_model, newx = test_matrix, type = "response")[,1]

# Calculate metrics
metrics_lasso <- calculate_metrics(lasso_pred_probs, test_Y)

# Assign values
a4 <- metrics_lasso$misclass_no
b4 <- metrics_lasso$misclass_yes
c4 <- metrics_lasso$auc

# Print results for Method 4
cat("Method 4: L1 Penalty (Lasso) using glmnet\n")
cat("Number of predictors (d4):", d4, "\n")
cat("Misclassified 'No':", a4, "\n")
cat("Misclassified 'Yes':", b4, "\n")
cat("AUC:", c4, "\n\n")
