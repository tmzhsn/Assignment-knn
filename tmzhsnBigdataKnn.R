# Install packages if not installed
install.packages("class")
install.packages("caret")
install.packages("e1071")

library(class)
library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(viridis)


# ------------------------
# 1. PREPROCESSING
# ------------------------

# Convert categorical variables to factors
df$shape <- as.factor(df$shape)
df$color <- as.factor(df$color)
df$taste <- as.factor(df$taste)
df$fruit_name <- as.factor(df$fruit_name)

# Train-test split
set.seed(123)
split <- createDataPartition(df$fruit_name, p = 0.8, list = FALSE)
train <- df[split, ]
test <- df[-split, ]

# Normalize numeric columns
pre <- preProcess(train[, c("size..cm.", "weight..g.", "avg_price....")],
                  method = c("center", "scale"))

train_norm <- predict(pre, train)
test_norm <- predict(pre, test)

# KNN MODEL

knn_pred <- knn(
  train = train_norm[, c("size..cm.", "weight..g.", "avg_price....")],
  test  = test_norm[, c("size..cm.", "weight..g.", "avg_price....")],
  cl    = train_norm$fruit_name,
  k = 5
)

cat("\n\n=== KNN RESULT ===\n")
print(confusionMatrix(knn_pred, test_norm$fruit_name))


cm_knn <- confusionMatrix(knn_pred, test_norm$fruit_name)$table
cm_knn_df <- as.data.frame(cm_knn)
colnames(cm_knn_df) <- c("Prediction","Reference","Count")

cm_knn_df <- cm_knn_df %>%
  mutate(label = ifelse(Count==0, "", as.character(Count)),
         text_col = ifelse(Count > max(Count)*0.45, "white", "black"))

cm_knn_df$Reference  <- factor(cm_knn_df$Reference,  levels = rev(sort(unique(as.character(cm_knn_df$Reference)))))
cm_knn_df$Prediction <- factor(cm_knn_df$Prediction, levels = rev(levels(cm_knn_df$Reference)))

p_knn <- ggplot(cm_knn_df, aes(x = Reference, y = Prediction, fill = Count)) +
  geom_tile(color = "grey70") +
  geom_text(aes(label = label, color = text_col), size = 3) +
  scale_color_identity() +
  scale_fill_viridis_c(option = "C") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Reference", y = "Prediction", title = "KNN Confusion Matrix Heatmap")

print(p_knn)





