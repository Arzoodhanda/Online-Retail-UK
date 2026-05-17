# ----------------------------
# **1. Load Libraries**
# ----------------------------
library(tidyverse)
library(lubridate)
library(caret)
library(rpart)
library(rpart.plot)
library(class)
library(cluster)
library(factoextra)
library(skimr)

# ----------------------------
# **2. Load Data with Column Specification**
# ----------------------------
# Define column types explicitly to prevent parsing issues
column_types <- cols(
  Invoice = col_character(),
  StockCode = col_character(),
  Description = col_character(),
  Quantity = col_double(),
  InvoiceDate = col_datetime(format = "%d-%m-%Y %H:%M"),
  Price = col_double(),
  Customer_ID = col_character(),  # Keep as character to preserve leading zeros
  Country = col_character()
)

# Read data with specified column types
data <- read_csv("online_retail.csv", col_types = column_types)

# Quick data overview
skim(data)

# ----------------------------
# **3. Data Cleaning**
# ----------------------------
clean_data <- data %>%
  # Remove rows with missing Customer_ID (essential for customer analysis)
  drop_na(Customer_ID) %>%
  # Remove returns/refunds (negative quantities)
  filter(Quantity > 0) %>%
  # Remove zero/negative prices
  filter(Price > 0) %>%
  # Remove duplicate rows
  distinct() %>%
  # Create TotalSpend feature
  mutate(TotalSpend = Quantity * Price) %>%
  # Convert Customer_ID to factor
  mutate(Customer_ID = as.factor(Customer_ID))

# Check cleaned data structure
glimpse(clean_data)

# ----------------------------
# **4. RFM Feature Engineering**
# ----------------------------
rfm_data <- clean_data %>%
  group_by(Customer_ID) %>%
  summarise(
    Recency = as.numeric(difftime(now(), max(InvoiceDate), units = "days")),
    Frequency = n_distinct(Invoice),
    Monetary = sum(TotalSpend),
    AvgOrderValue = mean(TotalSpend),
    ProductVariety = n_distinct(StockCode)
  ) %>%
  ungroup()

# Remove customers with only one purchase
rfm_data <- rfm_data %>% 
  filter(Frequency > 1)

# ----------------------------
# **5. Data Preprocessing for Modeling**
# ----------------------------
# Scale features (important for distance-based algorithms)
preProc <- preProcess(rfm_data[, -1], method = c("center", "scale"))
rfm_scaled <- predict(preProc, rfm_data)

# ----------------------------
# **6. Customer Segmentation (K-Means)**
# ----------------------------
# Determine optimal number of clusters
set.seed(123)
fviz_nbclust(rfm_scaled[, -1], kmeans, method = "wss") + 
  ggtitle("Elbow Method for Optimal Clusters")

# Fit K-Means (assuming 4 clusters based on elbow method)
kmeans_model <- kmeans(rfm_scaled[, -1], centers = 4, nstart = 25)

# Assign segments
rfm_data$Segment <- as.factor(kmeans_model$cluster)

# Label segments based on business understanding
segment_labels <- c("Low-Value Occasional", "Mid-Value Regular", 
                    "High-Value Loyal", "New High-Spenders")
rfm_data$Segment <- factor(rfm_data$Segment, labels = segment_labels)

# Visualize segments
fviz_cluster(kmeans_model, data = rfm_scaled[, -1], 
             geom = "point", main = "Customer Segments")

# ----------------------------
# **7. Classification Modeling**
# ----------------------------
# Split data (80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(rfm_data$Segment, p = 0.8, list = FALSE)
trainData <- rfm_data[trainIndex, ]
testData <- rfm_data[-trainIndex, ]

# **A. K-Nearest Neighbors (KNN)**
# ----------------------------
# Train KNN model
knn_pred <- knn(
  train = trainData[, c("Recency", "Frequency", "Monetary")],
  test = testData[, c("Recency", "Frequency", "Monetary")],
  cl = trainData$Segment,
  k = 5
)

# Evaluate KNN
knn_cm <- confusionMatrix(knn_pred, testData$Segment)
print(knn_cm)

# **B. Decision Tree**
# ----------------------------
# Train Decision Tree with cross-validation
tree_model <- train(
  Segment ~ Recency + Frequency + Monetary,
  data = trainData,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Visualize tree
prp(tree_model$finalModel, extra = 104, nn = TRUE,
    box.col = c("pink", "palegreen3", "lightblue", "gold")[tree_model$finalModel$frame$yval])

# Evaluate Decision Tree
tree_pred <- predict(tree_model, testData)
tree_cm <- confusionMatrix(tree_pred, testData$Segment)
print(tree_cm)

# ----------------------------
# **8. Save Outputs**
# ----------------------------
# Save segmented data
write_csv(rfm_data, "customer_segments_rfm.csv")

# Save models
saveRDS(knn_model, "knn_segmentation_model.rds")
saveRDS(tree_model, "decision_tree_segmentation_model.rds")

# Save preprocessing object
saveRDS(preProc, "preprocessing_object.rds")