library(ggplot2)
library(patchwork)

################################################################################
# FITNESS
cold = read.csv("COLD_START.csv", header = TRUE)
warm = read.csv("WARM_START.csv", header = TRUE)

# BOXPLOTS
# Generate dataframe per testing variable
df_mean <- data.frame(
  Fitness = c(cold$fit_mean, warm$fit_mean),
  Condition = factor(rep(c("Cold", "Warm"), each = length(cold$fit_mean)))
)

df_max <- data.frame(
  Fitness = c(cold$fit_max, warm$fit_max),
  Condition = factor(rep(c("Cold", "Warm"), each = length(cold$fit_max)))
)

warm_max_trim = sort(warm$fit_max)[0:8]
cold_max_trim = sort(cold$fit_max)[0:9]

df_max_trim <- data.frame(
  Fitness = c(cold_max_trim, warm_max_trim),
  Condition = factor(c(rep("Cold", 9), rep("Warm", 8)))
)

p1 = ggplot(df_mean, aes(x = Condition, y = Fitness, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Mean average fitness",
       x = "Initialization",
       y = "Fitness") +
  theme_minimal() +
  scale_fill_manual(values = c("Cold" = "skyblue", "Warm" = "brown2"))

p2 = ggplot(df_max, aes(x = Condition, y = Fitness, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Max average fitness",
       x = "Initialization",
       y = "Mean Fitness") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "skyblue", "Warm" = "brown2"))

p3 = ggplot(df_max_trim, aes(x = Condition, y = Fitness, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Max average fitness (trimmed)",
       x = "Initialization",
       y = "Mean Fitness") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "skyblue", "Warm" = "brown2"))

p1 + p2 + p3

# STATISTICAL TESTING
# Confirm normality
shapiro.test(cold$fit_mean)
shapiro.test(warm$fit_mean)
shapiro.test(cold$fit_max)
shapiro.test(warm$fit_max)

shapiro.test(warm_max_trim)
shapiro.test(cold_max_trim)

par(mfrow=c(2,3))
qqnorm(cold$fit_mean, main = "Cold: mean")
qqnorm(warm$fit_mean, main = "Warm: mean")
qqnorm(cold$fit_max, main = "Cold: max")
qqnorm(warm$fit_max, main = "Warm: max")
qqnorm(cold_max_trim, main = "Cold: max (trimmed)")
qqnorm(warm_max_trim, main = "Warm: max (trimmed)")

# NOT NORMAL ->  Use Wilcoxon signed rank test
wilcox.test(cold$fit_mean, warm$fit_mean, 
            paired = FALSE, alternative = "less")
wilcox.test(cold$fit_max, warm$fit_max, 
            paired = FALSE, alternative = "less")
wilcox.test(cold_max_trim, warm_max_trim, 
            paired = FALSE, alternative = "less")

################################################################################
# MORPHOLOGY
library(ggplot2)
library(patchwork)

# DIVERSITY AND NOVELTY
div = read.csv("REPORT_DIVERSITY.csv", header = TRUE)
nov = read.csv("REPORT_NOVELTY.csv", header = TRUE)

# Confirm normality
par(mfrow=c(3,2))
qqnorm(div$warm, main = "Diversity: warm")
qqnorm(div$cold, main = "Diversity: cold")
qqnorm(nov$warm_short, main = "Novelty_short: warm")
qqnorm(nov$cold_short, main = "Novelty_short: cold")
qqnorm(nov$warm_long, main = "Novelty_long: warm")
qqnorm(nov$cold_long, main = "Novelty_long: cold")

shapiro.test(div$warm)
shapiro.test(div$cold)
shapiro.test(nov$warm_short)
shapiro.test(nov$cold_short)
shapiro.test(nov$warm_long)
shapiro.test(nov$cold_long)

# t-tests
t.test(div$warm, div$cold, alternative="less")
t.test(nov$warm_short, nov$cold_short, alternative="less")
t.test(nov$warm_long, nov$cold_long, alternative="less")

# BOXPLOTS
# Diversity
df_div <- data.frame(
  Diversity = c(div$warm, div$cold),
  Condition = factor(rep(c("Warm", "Cold"), each = length(div$warm)))
)

# Short-term novelty
df_nov_short <- data.frame(
  Novelty = c(nov$warm_short, nov$cold_short),
  Condition = factor(rep(c("Warm", "Cold"), each = length(nov$warm_short)))
)

# Long-term novelty
df_nov_long <- data.frame(
  Novelty = c(nov$warm_long, nov$cold_long),
  Condition = factor(rep(c("Warm", "Cold"), each = length(nov$warm_long)))
)

# Plotting
p1 = ggplot(df_div, aes(x = Condition, y = Diversity, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Mean final diversity",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "darkturquoise", "Warm" = "deeppink"))

p2 = ggplot(df_nov_short, aes(x = Condition, y = Novelty, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Mean final short-term novelty",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "aquamarine2", "Warm" = "chocolate1"))

p3 = ggplot(df_nov_long, aes(x = Condition, y = Novelty, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Mean final long-term novelty",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() +
  scale_fill_manual(values = c("Cold" = "aquamarine2", "Warm" = "chocolate1"))

p1 + p2 + p3

# temp
data_warm <- data.frame(fit_max = warm$fit_max)
data_cold <- data.frame(fit_max = cold$fit_max)

# Define bin width (or use breaks via `cut`)
p1 = ggplot(data_warm, aes(x = fit_max)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "black") +
  scale_x_continuous(breaks = seq(1.0, 3.5, by = 0.1)) +
  labs(title = "Custom Histogram (ggplot2)", x = "Value", y = "Count") +
  geom_density(color = "red", size = 1.2) +
  theme_minimal() 

p2 = ggplot(data_cold, aes(x = fit_max)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "black") +
  scale_x_continuous(breaks = seq(1.0, 3.5, by = 0.1)) +
  labs(title = "Custom Histogram (ggplot2)", x = "Value", y = "Count") +
  geom_density(color = "red", size = 1.2) +
  theme_minimal() 
 p1 + p2
 