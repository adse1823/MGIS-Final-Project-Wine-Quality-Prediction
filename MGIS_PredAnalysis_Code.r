# ====================================
# Wine Quality Analysis and Prediction
# Using Random Forest and SVM Models
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
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)        # For SVM
library(pROC)
library(scales)
library(corrplot)
library(gridExtra)
library(ggplot2)
# Read and prepare data
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

