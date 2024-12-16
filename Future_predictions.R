# ====================================
# Predict 190 New Rows Based on Existing Features
# ====================================

# Load necessary libraries
library(tidyverse)
library(randomForest)

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
