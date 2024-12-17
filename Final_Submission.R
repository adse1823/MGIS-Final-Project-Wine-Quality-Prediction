# ====================================
# Predict 190 New Rows Based on Existing Features
# ====================================
# Install all required packages
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("e1071")
install.packages("pROC")
install.packages("scales")
install.packages("corrplot")
install.packages("gridExtra")
install.packages("ggplot2")
# After installation, load the libraries
#library(Rtools)
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)        # For SVM
library(pROC)
library(scales)
library(corrplot)
library(gridExtra)
library(ggplot2)


# Load dataset
wine_data_1 <- read.csv("wineQT.csv")

# Add Row Numbers for tracking
wine_data_1$RowNumber <- seq_len(nrow(wine_data_1))

# Treat 'quality' as a categorical variable
wine_data_1$quality <- as.factor(wine_data_1$quality)

# Remove RowNumber from the training data
wine_data_1_no_rownum <- wine_data_1 %>% select(-RowNumber)

# List of features to predict
feature_columns <- colnames(wine_data_1_no_rownum)[!colnames(wine_data_1_no_rownum) %in% c("quality")]

# Train Random Forest model using all 1143 rows
rf_model <- randomForest(quality ~ ., data = wine_data_1_no_rownum, ntree = 500)

# Simulate 190 new rows with random feature values (you can adjust the range or sampling method)
set.seed(123)
new_data <- as.data.frame(lapply(wine_data_1_no_rownum[, feature_columns], function(x) {
  sample(min(x):max(x), 190, replace = TRUE)
}))

# Predict the 'quality' for the simulated new data
new_data_predictions <- predict(rf_model, new_data)

# Add the predictions to the new data
new_data$predicted_quality <- new_data_predictions

# Combine the new simulated data with the predictions (Row Number for tracking)
new_data$RowNumber <- seq(from = max(wine_data_1$RowNumber) + 1, by = 1, length.out = 190)

# Save the predicted new rows to a CSV file
write.csv(new_data, "predicted_new_data.csv", row.names = FALSE)

# Display the first few rows of the new predicted data
cat("\nPredicted New Data:\n")
print(head(new_data))


# ====================================
# Wine Quality Analysis and Prediction
# Using Random Forest and SVM Models
# ====================================


wine_data <- read.csv('WineQT.csv')

# Initial data exploration
str(wine_data)
summary(wine_data)
cat("\nMissing Values:", sum(is.na(wine_data)), "\n")

# Data cleaning and preparation
wine_data <- wine_data[!duplicated(wine_data),]
if("Id" %in% colnames(wine_data)) {
  wine_data <- wine_data %>% select(-Id)
}

# ====================================
# Exploratory Data Analysis
# ====================================

# Correlation analysis
correlation_matrix <- cor(wine_data[, !colnames(wine_data) %in% c("quality")])
png("correlation_plot.png", width = 800, height = 800)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", addCoef.col = "black", number.cex = 0.7)
dev.off()

# Quality distribution visualization
quality_dist <- ggplot(wine_data, aes(x = quality)) +
  geom_bar(fill = "steelblue", alpha = 0.8) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +  # Updated syntax
  labs(title = "Distribution of Wine Quality Scores",
       x = "Quality Score", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Display the quality distribution plot
print(quality_dist)

# Optionally, save the quality distribution plot
ggsave("quality_distribution.png", quality_dist, width = 8, height = 6)

# Convert quality to factor for classification
wine_data$quality <- as.factor(wine_data$quality)

# Split data for training and testing
set.seed(123)
train_index <- createDataPartition(wine_data$quality, p = 0.7, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

# Feature scaling
preprocess_params <- preProcess(train_data %>% select(-quality), 
                                method = c("center", "scale"))
train_data_scaled <- predict(preprocess_params, train_data)
test_data_scaled <- predict(preprocess_params, test_data)

# ====================================
# Model 1: Random Forest
# ====================================
set.seed(123)
rf_control <- trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE)

rf_model <- train(quality ~ ., 
                  data = train_data_scaled, 
                  method = "rf",
                  trControl = rf_control,
                  ntree = 1000,
                  importance = TRUE)

# Random Forest Predictions
rf_pred <- predict(rf_model, test_data_scaled)
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$quality)
rf_probs <- predict(rf_model, test_data_scaled, type = "prob")
rf_roc <- multiclass.roc(test_data$quality, rf_probs)
rf_auc <- auc(rf_roc)

# ====================================
# Model 2: Support Vector Machine
# ====================================
# Train SVM model
svm_model <- svm(quality ~ ., 
                 data = train_data_scaled,
                 kernel = "radial",
                 cost = 10,
                 gamma = 0.1)

# SVM Predictions
svm_pred <- predict(svm_model, test_data_scaled)

# Load caret again
library(caret)

# Create confusion matrices using caret
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$quality)
svm_conf_matrix <- confusionMatrix(svm_pred, test_data$quality)

# ====================================
# Comparative Analysis
# ====================================
# Compare model accuracies
model_comparison <- data.frame(
  Model = c("Random Forest", "SVM"),
  Accuracy = c(rf_conf_matrix$overall["Accuracy"],
               svm_conf_matrix$overall["Accuracy"]),
  Kappa = c(rf_conf_matrix$overall["Kappa"],
            svm_conf_matrix$overall["Kappa"])
)

# ====================================
# Visualizations
# ====================================
# 1. Quality Distribution
quality_dist <- ggplot(wine_data, aes(x = quality)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Wine Quality",
       x = "Quality Score",
       y = "Count") +
  theme_minimal()

# 2. Model Comparison Plot
comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Model Accuracy Comparison",
       x = "Model",
       y = "Accuracy") +
  theme_minimal()

# 3. Prediction Comparison
prediction_comparison <- data.frame(
  Actual = as.numeric(as.character(test_data$quality)),
  RF_Predicted = as.numeric(as.character(rf_pred)),
  SVM_Predicted = as.numeric(as.character(svm_pred))
)

# ====================================
# Export Results for Visualization
# ====================================
# Combined predictions using base R
combined_predictions <- data.frame(
  Actual_Quality = test_data$quality,
  RF_Predicted = rf_pred,
  SVM_Predicted = svm_pred
)
# Add other features
other_features <- test_data[, !colnames(test_data) %in% c("quality")]
combined_predictions <- cbind(combined_predictions, other_features)

# Feature importance from Random Forest
feature_importance <- varImp(rf_model)$importance
feature_importance <- data.frame(
  Feature = rownames(feature_importance),
  Importance = feature_importance[,1]
)

# Model performance metrics
model_metrics <- model_comparison

# Save results for Tableau/Power BI
write.csv(combined_predictions, "wine_predictions_combined.csv", row.names = FALSE)
write.csv(feature_importance, "feature_importance.csv", row.names = FALSE)
write.csv(model_metrics, "model_metrics.csv", row.names = FALSE)

# ====================================
# Print Summary Results
# ====================================
cat("\nRandom Forest Performance:\n")
print(rf_conf_matrix$overall)
cat("\nRandom Forest AUC:", rf_auc, "\n")

cat("\nSVM Performance:\n")
print(svm_conf_matrix$overall)

cat("\nModel Comparison:\n")
print(model_comparison)

# Display plots
print(quality_dist)
print(comparison_plot)

# ====================================
# Additional Analysis for Tableau
# ====================================
# Create detailed prediction analysis using base R
prediction_analysis <- data.frame(
  Actual_Quality = unique(combined_predictions$Actual_Quality),
  RF_Accuracy = tapply(combined_predictions$RF_Predicted == combined_predictions$Actual_Quality, 
                       combined_predictions$Actual_Quality, mean),
  SVM_Accuracy = tapply(combined_predictions$SVM_Predicted == combined_predictions$Actual_Quality, 
                        combined_predictions$Actual_Quality, mean)
)



# Add Count column
prediction_analysis$Count <- table(combined_predictions$Actual_Quality)

# Export the analysis
write.csv(prediction_analysis, "prediction_analysis.csv", row.names = FALSE)



# ====================================
# Predict future wine quality
# ====================================

# Read the new data (without the 'quality' column)
new_data <- read.csv("predicted_new_data.csv")

# Ensure the data is scaled using the same preprocessing steps as the training data
new_data_scaled <- predict(preprocess_params, new_data)

# Predict using Random Forest
rf_quality_pred <- predict(rf_model, new_data_scaled)

# Predict using Support Vector Machine
svm_quality_pred <- predict(svm_model, new_data_scaled)

# Combine the predictions into a dataframe for output
predictions_combined <- data.frame(
  Row_Number = 1:nrow(new_data),  # Add row numbers
  RF_Predicted_Quality = rf_quality_pred,
  SVM_Predicted_Quality = svm_quality_pred
)

# Optional: Save the predictions to a new CSV file
write.csv(predictions_combined, "predicted_quality_future.csv", row.names = FALSE)

# Print the combined predictions
print(predictions_combined)






