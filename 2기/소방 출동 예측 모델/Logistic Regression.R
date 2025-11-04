#로지스틱2



# 1. 파일 경로 설정 및 데이터 불러오기
file_path <- "C:/Users/김나현/OneDrive/Desktop/BDAS/data3.csv"
data <- read.csv(file_path, fileEncoding = "UTF-8")

# 2. 결측값 처리 (평균으로 대체)
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- lapply(data[numeric_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# 3. "출동소방서명"에서 "용인 소방서"와 "화성소방서" 제외
data <- subset(data, !(출동소방서명 %in% c("용인 소방서", "화성소방서")))

# 4. 범주형 변수를 팩터로 변환
categorical_vars <- c("출동소방서명", "발생장소시군구명", "사고장소구분명", "사고원인_범주", "처리결과구분명")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# 5. "현장거리.km."를 이진형 변수로 변환
data$현장거리_binary <- ifelse(data$현장거리.km. >= 10, 1, 0)


# caret 패키지 설치 및 로드
if (!require(caret)) install.packages("caret")
library(caret)

#6.수정2
# 6-1. 현장거리_binary의 값 분포 확인
table(data$현장거리_binary)

# 6-2. 클래스 값이 모두 0일 경우 기준을 재설정
if (all(data$현장거리_binary == 0)) {
  # 예: 현장거리.km. 값이 5km 이상이면 1, 그렇지 않으면 0
  data$현장거리_binary <- ifelse(data$현장거리.km. >= 5, 1, 0)
}

# 6-3. 데이터의 클래스 비율 확인
table(data$현장거리_binary)

# 6-4. 데이터 분할 (createDataPartition 사용)
if (length(unique(data$현장거리_binary)) > 1) {
  set.seed(123)  # 재현성 유지
  train_index <- createDataPartition(data$현장거리_binary, p = 0.8, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
} else {
  # 6-5. 클래스 값이 단일한 경우, 랜덤 샘플링으로 데이터 분할
  set.seed(123)
  train_index <- sample(1:nrow(data), size = 0.8 * nrow(data))
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
}




# 7. One-hot 인코딩을 적용해 범주형 변수를 처리
if (!require(fastDummies)) install.packages("fastDummies")
library(fastDummies)

# 7-1. 훈련 데이터와 테스트 데이터에 동일하게 더미 변수 생성
train_data_dummy <- dummy_cols(train_data, select_columns = categorical_vars, 
                               remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_data_dummy <- dummy_cols(test_data, select_columns = categorical_vars, 
                              remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# 7-2. 훈련 데이터와 테스트 데이터에 공통 변수만 남기기
train_vars <- colnames(train_data_dummy)
test_vars <- colnames(test_data_dummy)

# 7-3. 테스트 데이터에 없는 훈련 데이터 변수를 추가 (값은 0으로 채움)
missing_vars <- setdiff(train_vars, test_vars)
for (var in missing_vars) {
  test_data_dummy[[var]] <- 0
}

# 7-4. 테스트 데이터의 변수 순서를 훈련 데이터와 동일하게 맞춤
test_data_dummy <- test_data_dummy[, train_vars, drop = FALSE]

# 8. Lasso 모델 학습
if (!require(glmnet)) install.packages("glmnet")
library(glmnet)

# 8-1. 모델 학습용 데이터 준비
x_train <- as.matrix(train_data_dummy[, -which(names(train_data_dummy) == "현장거리_binary")])
y_train <- train_data_dummy$현장거리_binary

# 8-2. Lasso 모델 학습
lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

# 8-3. 최적의 람다 값
best_lambda <- lasso_model$lambda.min
cat("최적의 람다 값:", best_lambda, "\n")



# 최적의 람다에서 모델의 계수 확인
coefficients <- coef(lasso_model, s = "lambda.min")

# S4 객체를 일반 행렬로 변환
coefficients_matrix <- as.matrix(coefficients)

# 선택된 변수 확인 (계수가 0이 아닌 변수)
selected_features <- rownames(coefficients_matrix)[coefficients_matrix != 0]

# 결과 출력
cat("선택된 변수:\n")
print(selected_features)





# 9. 테스트 데이터에 대한 예측
x_test <- as.matrix(test_data_dummy[, -which(names(test_data_dummy) == "현장거리_binary")])
lasso_predictions <- predict(lasso_model, s = best_lambda, newx = x_test, type = "response")

# 10. 예측 결과 출력
print(lasso_predictions)

# 11. 모델 성능 평가
# 예측 확률을 이진 분류로 변환
lasso_class_pred <- ifelse(lasso_predictions > 0.5, 1, 0)

# 성능 평가 (Confusion Matrix)
confusion_matrix <- table(Predicted = lasso_class_pred, Actual = test_data_dummy$현장거리_binary)
print(confusion_matrix)

# 성능 지표 (Accuracy)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")





# 계수를 데이터프레임으로 변환 (변수 이름과 계수 값 포함)
coefficients_df <- data.frame(
  변수 = rownames(coefficients_matrix),
  계수 = coefficients_matrix[, 1]
)

# 계수가 0이 아닌 변수만 필터링
coefficients_df <- coefficients_df[coefficients_df$계수 != 0, ]

# 계수를 절댓값 기준으로 정렬 (중요한 순서대로)
coefficients_df <- coefficients_df[order(abs(coefficients_df$계수), decreasing = TRUE), ]

# ggplot2 패키지 설치 및 로드
if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# 중요 변수 시각화
ggplot(coefficients_df, aes(x = reorder(변수, 계수), y = 계수)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "라소 회귀에서 선택된 변수 중요도",
    x = "변수",
    y = "계수 값"
  ) +
  theme_minimal()




