# 필요한 라이브러리 로드
library(readr)        # CSV 파일 읽기
library(xgboost)      # XGBoost 모델
library(caret)        # 데이터 분할 및 평가
library(Matrix)       # 희소행렬 생성
install.packages("corrplot")
library(corrplot)     # 상관행렬 시각화

# 1. 파일 경로 설정 및 데이터 불러오기 ----
file_path <- "C:/Users/김나현/OneDrive/Desktop/BDAS/data3.csv"
data <- read.csv(file_path, fileEncoding = "UTF-8")

# 2. 결측값 처리 (평균으로 대체) ----
# 숫자형 변수의 결측값을 평균으로 대체
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- lapply(data[numeric_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# 3. "출동소방서명"에서 "용인 소방서"와 "화성소방서" 제외 ----
data <- subset(data, !(출동소방서명 %in% c("용인 소방서", "화성소방서")))

# 4. 범주형 변수를 팩터로 변환 ----
categorical_vars <- c("출동소방서명", "발생장소시군구명", "사고장소구분명", "사고원인_범주", "처리결과구분명")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# 5. 상관관계 분석 (EDA) ----
# 연속형 변수만 선택
numeric_data <- data[, sapply(data, is.numeric)]

# 표준편차가 0인 열 제거
numeric_data <- numeric_data[, sapply(numeric_data, function(x) sd(x, na.rm = TRUE) > 0)]

# 결측값 및 Inf/NaN 처리
numeric_data <- as.data.frame(lapply(numeric_data, function(x) {
  x[is.infinite(x)] <- NA
  x[is.nan(x)] <- NA
  return(x)
}))

# 결측값 제거
numeric_data <- na.omit(numeric_data)

# 상관행렬 계산
cor_matrix <- cor(numeric_data, use = "complete.obs")

# 히트맵 생성
heatmap(
  cor_matrix,
  main = "Correlation Matrix",
  col = colorRampPalette(c("blue", "white", "red"))(100),
  margins = c(5, 5),
  distfun = function(x) as.dist(1 - x),
  hclustfun = function(x) hclust(x, method = "average"),
  cexRow = 1.0,              # 행 레이블 크기 조정
  cexCol = 1.0               # 열 레이블 크기 조정
)



# 6. 데이터 분할 (훈련/테스트 데이터셋) ----
set.seed(123)
train_index <- createDataPartition(data$현장거리.km., p = 0.8, list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]

# 7. XGBoost 모델 준비 ----
# 원-핫 인코딩 적용
train_matrix <- xgb.DMatrix(data = model.matrix(~ . - 1, data = train[, !colnames(train) %in% c("현장거리.km.")]),
                            label = train$현장거리.km.)
test_matrix <- model.matrix(~ . - 1, data = test[, !colnames(test) %in% c("현장거리.km.")])

# 훈련 데이터와 테스트 데이터의 특성 이름 동기화
train_features <- colnames(train_matrix)
test_features <- colnames(test_matrix)
missing_features <- setdiff(train_features, test_features)

for (feature in missing_features) {
  test_matrix <- cbind(test_matrix, 0)
  colnames(test_matrix)[ncol(test_matrix)] <- feature
}

test_matrix <- test_matrix[, train_features, drop = FALSE]
test_matrix <- xgb.DMatrix(data = test_matrix)

# 8. XGBoost 모델 학습 ----
xgb_model <- xgboost(
  data = train_matrix,
  max.depth = 6,
  eta = 0.1,
  nrounds = 100,
  objective = "reg:squarederror",
  verbose = 0
)

# 9. 모델 성능 평가 ----
# 테스트 데이터 예측
predictions <- predict(xgb_model, test_matrix)

# 성능 평가 (RMSE, R^2)
rmse <- sqrt(mean((predictions - test$현장거리.km.)^2))
rsq <- cor(predictions, test$현장거리.km.)^2
cat("RMSE:", rmse, "\n")
cat("R-squared:", rsq, "\n")

# 10. 중요 변수 시각화 ----
importance <- xgb.importance(model = xgb_model)
print(importance)
xgb.plot.importance(importance)



# 중요도 순으로 내림차순 정렬하여 상위 20개 변수 추출
top_20_importance <- importance %>%
  arrange(desc(Gain)) %>%
  head(20)

# 상위 20개 변수 시각화
xgb.plot.importance(top_20_importance)













