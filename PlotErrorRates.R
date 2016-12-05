library(ggplot2)

Error <- read.table(
  header=TRUE, text='Type        Method Error
  1  Train    ProxgradHinge      .165
  2  Test     ProxgradHinge      .348
  3  Train    ProxgradLogistic   .207
  4  Test     ProxgradLogistic   .410
  5  Train    LeastSquares       .134
  6  Test     LeastSquares       .297
  7  Train    LogisticRegression .144
  8  Test     LogisticRegression .161
  9  Train    Lasso              .194
  10 Test     Lasso              .210')

Error$Method <- factor(Error$Method, levels=c("LogisticRegression", "Lasso", "LeastSquares", "ProxgradHinge", "ProxgradLogistic"))

ggplot(Error, aes(factor(Method), Error, fill = Type)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")
