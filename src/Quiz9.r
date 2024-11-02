# Load required library
library(e1071)

# Load required library
library(e1071)

# Read the data
url = "https://liangfgithub.github.io/Data/spam.txt"
spam = read.table(url)

names(spam)[ncol(spam)]="Y"
spam$Y = as.factor(spam$Y)

testID = c(1:100, 1901:1960)
spam.test=spam[testID, ]
spam.train=spam[-testID, ]

# Function to evaluate SVM
evaluate_svm <- function(cost_val) {
    svmfit = svm(Y ~., kernel="linear", data=spam.train, cost=cost_val)
    
    # Get number of support vectors
    n_sv <- length(svmfit$index)
    
    # Get training error
    train_pred <- predict(svmfit, spam.train)
    train_error <- sum(train_pred != spam.train$Y)
    
    # Get test error
    test_pred <- predict(svmfit, spam.test)
    test_error <- sum(test_pred != spam.test$Y)
    
    cat("\nResults for cost =", cost_val, ":\n")
    cat("Number of support vectors:", n_sv, "\n")
    cat("Training error:", train_error, "\n")
    cat("Test error:", test_error, "\n")
    cat("----------------------\n")
    
    return(c(n_sv, train_error, test_error))
}

# Function to evaluate SVM
evaluate_svm_gaussian <- function(cost_val) {
    svmfit = svm(Y ~., kernel="radial", data=spam.train, cost=cost_val)
    
    # Get number of support vectors
    n_sv <- length(svmfit$index)
    
    # Get training error
    train_pred <- predict(svmfit, spam.train)
    train_error <- sum(train_pred != spam.train$Y)
    
    # Get test error
    test_pred <- predict(svmfit, spam.test)
    test_error <- sum(test_pred != spam.test$Y)
    
    cat("\nResults for cost =", cost_val, ":\n")
    cat("Number of support vectors:", n_sv, "\n")
    cat("Training error:", train_error, "\n")
    cat("Test error:", test_error, "\n")
    cat("----------------------\n")
    
    return(c(n_sv, train_error, test_error))
}

# Evaluate for each cost value
costs <- c(1, 10, 50)
results <- sapply(costs, evaluate_svm)
results_gaussian <- sapply(costs, evaluate_svm_gaussian)

# Create a clean summary table
rownames(results) <- c("support_vectors", "train_error", "test_error")
colnames(results) <- paste0("cost_", costs)

cat("\nSummary of all results:\n")
print(results)