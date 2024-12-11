# Load necessary libraries
library(tidyverse)       # For data manipulation and visualization
library(caret)           # For model training and evaluation
library(randomForest)    # For Random Forest
library(pROC)            # For ROC curves and AUC calculation
library(e1071)           # For model tuning

# Set file path for dataset
file_path <- "C:/Users/lilbu/Documents/MGIS-Final-Project-Wine-Quality-Prediction/WineQT.csv"

# Load the dataset
wine_data <- read.csv(file_path)

# Explore the dataset
str(wine_data)  # Print the structure of the dataset
summary(wine_data)  # Print summary statistics of the dataset

# Check for missing values
sum(is.na(wine_data))  # Count missing values in the dataset

# Visualize the distribution of wine quality
ggplot(wine_data, aes(x = as.factor(quality))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Wine Quality", x = "Quality", y = "Count")  # Bar plot of wine quality distribution

# Convert 'quality' to factor for classification tasks
wine_data$quality <- as.factor(wine_data$quality)  # Convert quality to categorical variable

# Remove 'Id' column if present
if("Id" %in% colnames(wine_data)) {
  wine_data <- wine_data %>% select(-Id)  # Drop 'Id' column if it exists
}

# Split the dataset into training and testing sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(wine_data$quality, p = 0.7, list = FALSE)  # Create train-test split
train_data <- wine_data[train_index, ]  # Training data
test_data <- wine_data[-train_index, ]  # Testing data

# Feature scaling (normalization)
preprocess_params <- preProcess(train_data %>% select(-quality), method = c("center", "scale"))  # Scaling parameters
train_data_scaled <- predict(preprocess_params, train_data)  # Scale training data
test_data_scaled <- predict(preprocess_params, test_data %>% select(-quality))  # Scale testing data

# Train Random Forest model with Cross-validation
set.seed(123)  # Set seed for reproducibility
rf_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)  # 10-fold cross-validation
rf_model <- train(quality ~ ., data = train_data_scaled, method = "rf", trControl = rf_control, ntree = 1000, importance = TRUE)  # Train Random Forest model
print(rf_model)  # Print model summary

# Evaluate Random Forest model
rf_pred <- predict(rf_model, test_data_scaled)  # Predict on test data
confusionMatrix(rf_pred, test_data$quality)  # Print confusion matrix

# Visualize feature importance for Random Forest
varImpPlot(rf_model$finalModel, main = "Feature Importance (Random Forest)")  # Feature importance plot

# Compare model performance
rf_accuracy <- confusionMatrix(rf_pred, test_data$quality)$overall['Accuracy']  # Calculate accuracy
cat("Random Forest Accuracy:", rf_accuracy, "\n")  # Print accuracy

# Plot ROC curve for the Random Forest model
rf_probs <- predict(rf_model, test_data_scaled, type = "prob")  # Predict probabilities
r <- multiclass.roc(test_data$quality, rf_probs)  # Compute ROC for multiclass
auc_value <- auc(r)  # Calculate AUC

# Display AUC for the model
cat("Random Forest AUC:", auc_value, "\n")  # Print AUC

# Suggestion for teammates:
# Consider adding models such as SVM, k-NN, or Logistic Regression using the same workflow.
# Use the same train-test split and scaling process to ensure comparability.
# Evaluate additional models using cross-validation and compare results using accuracy, AUC, and feature importance.
