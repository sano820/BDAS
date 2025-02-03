library(dplyr)
library()

data1 <- read.csv("C:/Users/박상선의 노트북/OneDrive/바탕 화면/출동 예측/data_부천.csv", fileEncoding = "UTF-8")
data2 <- read.csv("C:/Users/박상선의 노트북/OneDrive/바탕 화면/출동 예측/data_안산.csv", fileEncoding = "UTF-8")
data3 <- read.csv("C:/Users/박상선의 노트북/OneDrive/바탕 화면/출동 예측/data_안양.csv", fileEncoding = "UTF-8")


# 수치데이터 평균 처리
filtered_data <- data1 %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# 범주형 더미변수화
data_for_dummy <- filtered_data %>% select(-현장거리.km.)
dummy_data <- dummy_cols(data_for_dummy, 
                         remove_first_dummy = TRUE, 
                         remove_selected_columns = TRUE)

# 더미 변수화된 데이터와 현장거리.km. 병합
final_data <- cbind(dummy_data, 현장거리.km. = filtered_data$현장거리.km.)

# 데이터 나누기 (80% train, 20% test)
set.seed(123)
train_index <- createDataPartition(final_data$현장거리.km., p = 0.8, list = FALSE)
train_data <- final_data[train_index, ]
test_data <- final_data[-train_index, ]

# 다항 회귀분석
poly_model <- lm(현장거리.km. ~ ., data = train_data)
summary(poly_model)

# 예측 및 성능 평가 (다항 회귀)
poly_pred <- predict(poly_model, newdata = test_data)
poly_rmse <- sqrt(mean((poly_pred - test_data$현장거리.km.)^2))
cat("다항 회귀 RMSE: ", poly_rmse, "\n")



# 예측값과 실제값 비교
comparison <- data.frame(
  실제값 = test_data$현장거리.km.,
  예측값 = poly_pred
)

# 1. 실제값 vs 예측값 산점도
ggplot(comparison, aes(x = 실제값, y = 예측값)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("실제값 vs 예측값") +
  xlab("실제 현장거리.km.") +
  ylab("예측된 현장거리.km.") +
  theme_minimal()

# 2. 잔차 계산
comparison$residuals <- comparison$실제값 - comparison$예측값

# 3. 잔차 분포 확인 (히스토그램)
ggplot(comparison, aes(x = residuals)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  ggtitle("잔차 분포") +
  xlab("잔차값") +
  ylab("빈도") +
  theme_minimal()

# 4. 잔차 vs 예측값 플롯
ggplot(comparison, aes(x = 예측값, y = residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  ggtitle("잔차 vs 예측값") +
  xlab("예측된 현장거리.km.") +
  ylab("잔차값") +
  theme_minimal()

# 5. Q-Q 플롯 (잔차 정규성 확인)
qqnorm(comparison$residuals, main = "Q-Q Plot for Residuals")
qqline(comparison$residuals, col = "red", lwd = 2)

# 6. 모델의 변수 중요도 (절대값 기준)
coefficients <- summary(poly_model)$coefficients
coeff_importance <- data.frame(
  변수 = rownames(coefficients),
  계수 = coefficients[, 1],
  중요도 = abs(coefficients[, 1])
) %>% arrange(desc(중요도))

print(coeff_importance)

# 7. 중요 변수 시각화
ggplot(coeff_importance, aes(x = reorder(변수, 중요도), y = 중요도)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  coord_flip() +
  ggtitle("다항 회귀 모델 변수 중요도") +
  xlab("변수") +
  ylab("중요도 (절대값)") +
  theme_minimal()

# 상위 10% 중요도 변수 필터링
top_10_percent <- coeff_importance %>%
  filter(중요도 >= quantile(중요도, 0.9))

# 상위 10% 변수 중요도 시각화
ggplot(top_10_percent, aes(x = reorder(변수, 중요도), y = 중요도)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  coord_flip() +
  ggtitle("다항 회귀 모델 변수 중요도 (상위 10%)") +
  xlab("변수") +
  ylab("중요도 (절대값)") +
  theme_minimal()

# 중요도 상위 10개 변수 필터링
top_10_vars <- coeff_importance %>%
  arrange(desc(중요도)) %>%
  head(10)

# 상위 10개 변수 중요도 시각화
ggplot(top_10_vars, aes(x = reorder(변수, 중요도), y = 중요도)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  coord_flip() +
  ggtitle("다항 회귀 모델 변수 중요도 (상위 10개)") +
  xlab("변수") +
  ylab("중요도 (절대값)") +
  theme_minimal()

# 중요도 상위 20개 변수 필터링
top_20_vars <- coeff_importance %>%
  arrange(desc(중요도)) %>%
  head(20)

# 상위 20개 변수 중요도 시각화
ggplot(top_20_vars, aes(x = reorder(변수, 중요도), y = 중요도)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  coord_flip() +
  ggtitle("다항 회귀 모델 변수 중요도 (상위 20개)") +
  xlab("변수") +
  ylab("중요도 (절대값)") +
  theme_minimal()
