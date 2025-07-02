library(ggplot2)
library(patchwork)
library(dplyr)

################################################################################
# BOXPLOTS
# FEATURE MATRICES
cold = read.csv("morpho_cold.csv", header = TRUE)
warm = read.csv("morpho_warm.csv", header = TRUE)

df_cnt_brick <- data.frame(
  Count = c(warm$cnt_bricks, cold$cnt_bricks),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_cnt_joint <- data.frame(
  Count = c(warm$cnt_joint, cold$cnt_joint),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_vol_bbox <- data.frame(
  Volume = c(warm$vol_bbox, cold$vol_bbox),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_vol_disp <- data.frame(
  Volume = c(warm$vol_disp, cold$vol_disp),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_vol_bbox <- data.frame(
  Volume = c(warm$vol_bbox, cold$vol_bbox),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_cnt_limb <- data.frame(
  Count = c(warm$cnt_limb, cold$cnt_limb),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_len_limb <- data.frame(
  Length = c(warm$avg_len_limb, cold$avg_len_limb),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_nose <- data.frame(
  Nose = c(warm$nose, cold$nose),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_sym_x <- data.frame(
  Symmetry = c(warm$sym_x, cold$sym_x),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_sym_y <- data.frame(
  Symmetry = c(warm$sym_y, cold$sym_y),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

df_sym_z <- data.frame(
  Symmetry = c(warm$sym_z, cold$sym_z),
  Condition = factor(rep(c("Warm", "Cold"), each = 10))
)

# BOXPLOTS
p1 = ggplot(df_cnt_brick, aes(x = Condition, y = Count, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Brick count",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p2 = ggplot(df_cnt_joint, aes(x = Condition, y = Count, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Joint count",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p3 = ggplot(df_cnt_limb, aes(x = Condition, y = Count, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Limb count",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p4 = ggplot(df_len_limb, aes(x = Condition, y = Length, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Avg limb length",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p5 = ggplot(df_nose, aes(x = Condition, y = Nose, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Nose",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

################################################################################

p6 = ggplot(df_sym_x, aes(x = Condition, y = Symmetry, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Sym X",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p7 = ggplot(df_sym_y, aes(x = Condition, y = Symmetry, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Sym Y",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p8 = ggplot(df_sym_z, aes(x = Condition, y = Symmetry, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Sym Z",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p9 = ggplot(df_vol_bbox, aes(x = Condition, y = Volume, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Vol B-Box",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

p10 = ggplot(df_vol_disp, aes(x = Condition, y = Volume, fill = Condition)) +
  geom_boxplot() +
  labs(title = "Vol Disp",
       x = "Condition",
       y = "Diversity") +
  theme_minimal() + 
  scale_fill_manual(values = c("Cold" = "cornflowerblue", "Warm" = "brown2"))

################################################################################

wrap_plots(mget(paste0("p", 1:10)), nrow = 2, ncol = 5) +
plot_layout(guides = "collect")

#(p1 + p2 + p3 + p4 + p5) /
#(p6 + p7 + p8 + p9 + p10)

################################################################################
# STATISTICAL TESTING
# DEEPER FEATURE ANALYSIS
# To compare:
# - Joint count
# - Brick count
# - Displacement (total count)
# - Avg limb length
# - Limb count

# Determine test type -> Test for normality
# QQ-plots
par(mfrow=c(2,5))
# Warm start features
qqnorm(warm$cnt_bricks, main = "Warm: brick count")
qqnorm(warm$cnt_joints, main = "Warm: joint count")
qqnorm(warm$vol_disp, main = "Warm: displacement")
qqnorm(warm$cnt_limb, main = "Warm: limb count")
qqnorm(warm$avg_len_limb, main = "Warm: avg limb length")

# Cold start features
qqnorm(cold$cnt_bricks, main = "Cold: brick count")
qqnorm(cold$cnt_joints, main = "Cold: joint count")
qqnorm(cold$vol_disp, main = "Cold: displacement")
qqnorm(cold$cnt_limb, main = "Cold: limb count")
qqnorm(cold$avg_len_limb, main = "Cold: avg limb length")

# Normality might only be assumed for brick count -> test
shapiro.test(warm$cnt_bricks)
shapiro.test(cold$cnt_bricks)

# Indeed seems to be the case. Test the rest of the features using Wilcoxon tests
t.test(warm$cnt_bricks, cold$cnt_bricks, alternative = "greater")

wilcox.test(warm$cnt_joints, cold$cnt_joints, alternative = "less", exact = FALSE)
wilcox.test(warm$vol_disp, cold$vol_disp, alternative = "less", exact = FALSE)
wilcox.test(warm$avg_len_limb, cold$avg_len_limb, alternative = "less", exact = FALSE)
wilcox.test(warm$cnt_limb, cold$cnt_limb, alternative = "less", exact = FALSE)
wilcox.test(warm$nose, cold$nose, alternative = "greater", exact = FALSE)
