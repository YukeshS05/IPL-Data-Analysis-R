# ============================================
# IPL Data Analysis & Match Outcome Prediction
# Subject: R for Data Science
# ============================================

# Load all libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(readr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

# Set working directory (change path to match YOUR folder)
setwd("C:/Users/Yukesh/Documents/IPL_Project")  # Windows
# setwd("/Users/YourName/IPL_Project")  # Mac/Linux

# ============================================
# STEP 2: DATA LOADING & EXPLORATION
# ============================================

# Load the two datasets
matches <- read_csv("data/matches.csv")
deliveries <- read_csv("data/deliveries.csv")

# Confirm they loaded successfully
cat("Matches dataset loaded:", nrow(matches), "rows and", ncol(matches), "columns\n")
cat("Deliveries dataset loaded:", nrow(deliveries), "rows and", ncol(deliveries), "columns\n")

# See first 6 rows of each dataset
head(matches)
head(deliveries)

# See last 6 rows
tail(matches)
tail(deliveries)

# Structure: column names, data types, sample values
str(matches)
str(deliveries)

# Summary statistics for every column
summary(matches)
summary(deliveries)

# See all column names clearly
colnames(matches)
colnames(deliveries)

# Count missing values in each column
cat("\n--- Missing values in matches ---\n")
colSums(is.na(matches))

cat("\n--- Missing values in deliveries ---\n")
colSums(is.na(deliveries))

# How many seasons are in the data?
unique(matches$season)

# Which teams appear?
unique(matches$team1)

# How many unique venues?
length(unique(matches$venue))

# Toss decisions distribution
table(matches$toss_decision)

# How many matches per season?
table(matches$season)

# The 'id' in matches should match 'match_id' in deliveries
cat("Match IDs in matches:", length(unique(matches$id)), "\n")
cat("Match IDs in deliveries:", length(unique(deliveries$match_id)), "\n")

# ============================================
# STEP 3: DATA CLEANING
# ============================================

# Fix season: extract the first 4 digits as the starting year
matches <- matches %>%
  mutate(season = as.integer(substr(season, 1, 4)))

# Verify the fix
unique(matches$season)

# Fix all inconsistent team names across both datasets
fix_team_names <- function(team) {
  team <- case_when(
    team == "Rising Pune Supergiant"      ~ "Rising Pune Supergiants",
    team == "Royal Challengers Bangalore" ~ "Royal Challengers Bengaluru",
    team == "Kings XI Punjab"             ~ "Punjab Kings",
    team == "Delhi Daredevils"            ~ "Delhi Capitals",
    TRUE                                  ~ team  # keep everything else as-is
  )
  return(team)
}

# Apply to matches dataset
matches <- matches %>%
  mutate(
    team1        = fix_team_names(team1),
    team2        = fix_team_names(team2),
    toss_winner  = fix_team_names(toss_winner),
    winner       = fix_team_names(winner)
  )

# Apply to deliveries dataset
deliveries <- deliveries %>%
  mutate(
    batting_team = fix_team_names(batting_team),
    bowling_team = fix_team_names(bowling_team)
  )

# Verify - should now show clean unique team names
unique(matches$team1)


# Remove matches where there is no winner (rained out / no result)
cat("Rows before removing no-result matches:", nrow(matches), "\n")

matches <- matches %>%
  filter(!is.na(winner))

cat("Rows after removing no-result matches:", nrow(matches), "\n")


# Check which venues have missing cities
matches %>%
  filter(is.na(city)) %>%
  select(venue, city) %>%
  distinct()

# Fill missing city from venue name (manual mapping)
matches <- matches %>%
  mutate(city = case_when(
    is.na(city) & str_detect(venue, "Dubai")       ~ "Dubai",
    is.na(city) & str_detect(venue, "Sharjah")     ~ "Sharjah",
    is.na(city) & str_detect(venue, "Abu Dhabi")   ~ "Abu Dhabi",
    is.na(city) & str_detect(venue, "Cape Town")   ~ "Cape Town",
    is.na(city) & str_detect(venue, "Centurion")   ~ "Centurion",
    is.na(city) & str_detect(venue, "Johannesburg") ~ "Johannesburg",
    is.na(city) & str_detect(venue, "Durban")      ~ "Durban",
    is.na(city) & str_detect(venue, "Port Elizabeth") ~ "Port Elizabeth",
    is.na(city) & str_detect(venue, "Kimberley")   ~ "Kimberley",
    is.na(city) & str_detect(venue, "Bloemfontein") ~ "Bloemfontein",
    TRUE ~ city
  ))

# Check remaining NAs in city
sum(is.na(matches$city))


# Drop 'method' column - 98% empty, not useful
# Drop 'umpire1', 'umpire2' - not relevant for match outcome prediction
matches <- matches %>%
  select(-method, -umpire1, -umpire2)

# Verify remaining columns
colnames(matches)


# Over goes 0-19, convert to 1-20 for readability
deliveries <- deliveries %>%
  mutate(over = over + 1)

# Verify
range(deliveries$over)
# Should show: [1]  1 20


# Add a useful new column: did the toss winner also win the match?
matches <- matches %>%
  mutate(toss_winner_won = ifelse(toss_winner == winner, "Yes", "No"))

# Quick check
table(matches$toss_winner_won)


# Final missing value check
cat("\n=== FINAL MISSING VALUE CHECK ===\n")
cat("\n--- matches ---\n")
colSums(is.na(matches))

cat("\n--- deliveries ---\n")
colSums(is.na(deliveries))

# Final dataset dimensions
cat("\nFinal matches shape:", nrow(matches), "rows,", ncol(matches), "cols\n")
cat("Final deliveries shape:", nrow(deliveries), "rows,", ncol(deliveries), "cols\n")

# Final season check
cat("\nMatches per season after cleaning:\n")
table(matches$season)

# Final team names check
cat("\nUnique teams after cleaning:\n")
sort(unique(c(matches$team1, matches$team2)))


# Save cleaned datasets so we don't have to re-clean every time
write_csv(matches, "data/matches_clean.csv")
write_csv(deliveries, "data/deliveries_clean.csv")

cat("✅ Cleaned datasets saved successfully!\n")


# ============================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================

# Chart 1: Total wins by team
team_wins <- matches %>%
  group_by(winner) %>%
  summarise(wins = n()) %>%
  arrange(desc(wins))

ggplot(team_wins, aes(x = reorder(winner, wins), y = wins, fill = wins)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "#56B4E9", high = "#E69F00") +
  labs(
    title = "Total IPL Wins by Team (2008–2024)",
    x = "Team",
    y = "Number of Wins",
    fill = "Wins"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plots/01_total_wins_by_team.png", width = 10, height = 6)
cat("Chart 1 saved!\n")


# Chart 2: Toss decision (bat vs field) across seasons
toss_trend <- matches %>%
  group_by(season, toss_decision) %>%
  summarise(count = n(), .groups = "drop")

ggplot(toss_trend, aes(x = season, y = count, color = toss_decision, group = toss_decision)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("bat" = "#E69F00", "field" = "#0072B2")) +
  labs(
    title = "Toss Decision Trend Across IPL Seasons",
    x = "Season",
    y = "Number of Matches",
    color = "Toss Decision"
  ) +
  theme_minimal(base_size = 13)

ggsave("plots/02_toss_decision_trend.png", width = 10, height = 6)
cat("Chart 2 saved!\n")


# Chart 3: Toss winner vs match winner
toss_impact <- matches %>%
  group_by(toss_winner_won) %>%
  summarise(count = n()) %>%
  mutate(percentage = round(count / sum(count) * 100, 1))

ggplot(toss_impact, aes(x = toss_winner_won, y = count, fill = toss_winner_won)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(percentage, "%")), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Yes" = "#009E73", "No" = "#D55E00")) +
  labs(
    title = "Does Winning the Toss Lead to Winning the Match?",
    x = "Toss Winner Also Won the Match?",
    y = "Number of Matches",
    fill = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plots/03_toss_impact.png", width = 7, height = 6)
cat("Chart 3 saved!\n")


# Chart 4: How many matches won by runs (bat first) vs wickets (chase)?
win_type <- matches %>%
  filter(result %in% c("runs", "wickets")) %>%
  group_by(result) %>%
  summarise(count = n()) %>%
  mutate(
    label = ifelse(result == "runs", "Won Batting First", "Won Chasing"),
    percentage = round(count / sum(count) * 100, 1)
  )

ggplot(win_type, aes(x = "", y = count, fill = label)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(label, "\n", percentage, "%")),
            position = position_stack(vjust = 0.5), size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Won Batting First" = "#E69F00", "Won Chasing" = "#56B4E9")) +
  labs(title = "IPL Matches: Batting First vs Chasing Wins") +
  theme_void(base_size = 13) +
  theme(legend.position = "none")

ggsave("plots/04_batting_vs_chasing.png", width = 7, height = 6)
cat("Chart 4 saved!\n")


# Chart 5: Top 10 batsmen by total runs
top_batsmen <- deliveries %>%
  group_by(batter) %>%
  summarise(total_runs = sum(batsman_runs)) %>%
  arrange(desc(total_runs)) %>%
  slice(1:10)

ggplot(top_batsmen, aes(x = reorder(batter, total_runs), y = total_runs, fill = total_runs)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = total_runs), hjust = -0.1, size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#56B4E9", high = "#D55E00") +
  labs(
    title = "Top 10 Run Scorers in IPL History",
    x = "Batsman",
    y = "Total Runs",
    fill = "Runs"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  ylim(0, max(top_batsmen$total_runs) * 1.1)

ggsave("plots/05_top_batsmen.png", width = 10, height = 6)
cat("Chart 5 saved!\n")


# Chart 6: Top 10 bowlers by total wickets
top_bowlers <- deliveries %>%
  filter(is_wicket == 1) %>%
  filter(!dismissal_kind %in% c("run out", "retired hurt", "obstructing the field")) %>%
  group_by(bowler) %>%
  summarise(wickets = n()) %>%
  arrange(desc(wickets)) %>%
  slice(1:10)

ggplot(top_bowlers, aes(x = reorder(bowler, wickets), y = wickets, fill = wickets)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = wickets), hjust = -0.1, size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#56B4E9", high = "#009E73") +
  labs(
    title = "Top 10 Wicket Takers in IPL History",
    x = "Bowler",
    y = "Total Wickets",
    fill = "Wickets"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  ylim(0, max(top_bowlers$wickets) * 1.1)

ggsave("plots/06_top_bowlers.png", width = 10, height = 6)
cat("Chart 6 saved!\n")


# Chart 7: Number of matches played each season
season_matches <- matches %>%
  group_by(season) %>%
  summarise(total_matches = n())

ggplot(season_matches, aes(x = factor(season), y = total_matches, fill = total_matches)) +
  geom_col() +
  geom_text(aes(label = total_matches), vjust = -0.4, size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#56B4E9", high = "#E69F00") +
  labs(
    title = "Number of IPL Matches Played Per Season",
    x = "Season",
    y = "Total Matches"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

ggsave("plots/07_matches_per_season.png", width = 10, height = 6)
cat("Chart 7 saved!\n")


# Chart 8: Most used venues
top_venues <- matches %>%
  group_by(venue) %>%
  summarise(matches_hosted = n()) %>%
  arrange(desc(matches_hosted)) %>%
  slice(1:10)

ggplot(top_venues, aes(x = reorder(venue, matches_hosted),
                       y = matches_hosted, fill = matches_hosted)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = matches_hosted), hjust = -0.1, size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#56B4E9", high = "#CC79A7") +
  labs(
    title = "Top 10 IPL Venues by Matches Hosted",
    x = "Venue",
    y = "Matches Hosted"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  ylim(0, max(top_venues$matches_hosted) * 1.15)

ggsave("plots/08_top_venues.png", width = 11, height = 6)
cat("Chart 8 saved!\n")


# Print a clean summary of key findings
cat("\n========================================\n")
cat("       KEY EDA FINDINGS SUMMARY\n")
cat("========================================\n")

cat("\n🏆 Most Successful Team:\n")
print(head(team_wins, 3))

cat("\n🏏 Top 3 Run Scorers:\n")
print(head(top_batsmen, 3))

cat("\n🎯 Top 3 Wicket Takers:\n")
print(head(top_bowlers, 3))

cat("\n🎲 Toss Impact:\n")
print(toss_impact)

cat("\n🏟️ Most Used Venue:\n")
print(head(top_venues, 1))

cat("\n========================================\n")


# ============================================
# STEP 5: FEATURE ENGINEERING
# ============================================

# Load clean data (in case you're starting fresh)
matches <- read_csv("data/matches_clean.csv")
deliveries <- read_csv("data/deliveries_clean.csv")

cat("Clean data loaded successfully!\n")
cat("Matches:", nrow(matches), "rows\n")
cat("Deliveries:", nrow(deliveries), "rows\n")


# Calculate each team's overall win rate across all matches
team_win_rate <- matches %>%
  pivot_longer(cols = c(team1, team2), 
               names_to = "role", 
               values_to = "team") %>%
  group_by(team) %>%
  summarise(
    total_matches = n(),
    total_wins    = sum(winner == team),
    win_rate      = round(total_wins / total_matches, 3)
  ) %>%
  arrange(desc(win_rate))

# Preview
cat("\n--- Team Win Rates ---\n")
print(team_win_rate)


# Build feature dataset
model_data <- matches %>%
  mutate(
    # Feature 1: Did team1 win? (Our TARGET variable - what we want to predict)
    team1_wins = ifelse(winner == team1, 1, 0),
    
    # Feature 2: Did the toss winner choose to field? (Strongest toss strategy)
    toss_field = ifelse(toss_decision == "field", 1, 0),
    
    # Feature 3: Did team1 win the toss?
    team1_won_toss = ifelse(toss_winner == team1, 1, 0),
    
    # Feature 4: Is it a day-night or knockout match?
    is_playoff = ifelse(match_type %in% c("Final", "Qualifier 1", 
                                          "Qualifier 2", "Eliminator"), 1, 0),
    
    # Feature 5: Was it a super over match?
    is_super_over = ifelse(super_over == "Y", 1, 0)
  )

cat("Base features created!\n")


# Join win rates for team1
model_data <- model_data %>%
  left_join(team_win_rate %>% select(team, win_rate),
            by = c("team1" = "team")) %>%
  rename(team1_win_rate = win_rate)

# Join win rates for team2
model_data <- model_data %>%
  left_join(team_win_rate %>% select(team, win_rate),
            by = c("team2" = "team")) %>%
  rename(team2_win_rate = win_rate)

# Feature 6: Win rate difference (positive = team1 is stronger historically)
model_data <- model_data %>%
  mutate(win_rate_diff = round(team1_win_rate - team2_win_rate, 3))

cat("Win rate features added!\n")


# Feature 7: How many matches has this venue hosted? (bigger venues = more pressure)
venue_counts <- matches %>%
  group_by(venue) %>%
  summarise(venue_match_count = n())

model_data <- model_data %>%
  left_join(venue_counts, by = "venue")

cat("Venue feature added!\n")


# Get all unique team names and assign a number to each
all_teams <- sort(unique(c(matches$team1, matches$team2)))
cat("\nAll teams and their encoded numbers:\n")
print(data.frame(team_id = 1:length(all_teams), team_name = all_teams))

# Create encoding
team_encoding <- setNames(1:length(all_teams), all_teams)

# Apply encoding to model_data
model_data <- model_data %>%
  mutate(
    team1_encoded = team_encoding[team1],
    team2_encoded = team_encoding[team2]
  )

cat("\nTeam encoding applied!\n")


# Encode toss_decision: field = 1, bat = 0
model_data <- model_data %>%
  mutate(toss_decision_encoded = ifelse(toss_decision == "field", 1, 0))

cat("Toss decision encoded!\n")


# Select only the columns the model will use
model_features <- model_data %>%
  select(
    # Target variable
    team1_wins,
    
    # Match features
    team1_encoded,
    team2_encoded,
    team1_won_toss,
    toss_field,
    toss_decision_encoded,
    is_playoff,
    is_super_over,
    
    # Performance features
    team1_win_rate,
    team2_win_rate,
    win_rate_diff,
    
    # Venue feature
    venue_match_count
  ) %>%
  # Remove any rows with NA in these columns
  drop_na()

cat("\nFinal model features shape:", nrow(model_features), "rows,",
    ncol(model_features), "cols\n")


# For classification, the target must be a factor (category), not a number
model_features <- model_features %>%
  mutate(team1_wins = factor(team1_wins, levels = c(0, 1),
                             labels = c("Team2_Wins", "Team1_Wins")))

# Check the balance of our target variable
cat("\nTarget variable distribution:\n")
print(table(model_features$team1_wins))

cat("\nPercentage split:\n")
print(round(prop.table(table(model_features$team1_wins)) * 100, 1))


# Correlation heatmap of numeric features
numeric_features <- model_features %>%
  mutate(team1_wins_num = ifelse(team1_wins == "Team1_Wins", 1, 0)) %>%
  select(-team1_wins) %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_features, use = "complete.obs")

png("plots/09_correlation_heatmap.png", width = 900, height = 800, res = 100)
corrplot(cor_matrix,
         method   = "color",
         type     = "upper",
         tl.cex   = 0.85,
         tl.col   = "black",
         addCoef.col = "black",
         number.cex  = 0.7,
         title    = "Feature Correlation Matrix",
         mar      = c(0, 0, 2, 0))
dev.off()

cat("Chart 9 - Correlation heatmap saved!\n")


# Save the feature dataset for use in Step 6
write_csv(model_features, "data/model_features.csv")

cat("\n✅ Model features saved to data/model_features.csv\n")

# Final summary
cat("\n==========================================\n")
cat("     FEATURE ENGINEERING SUMMARY\n")
cat("==========================================\n")
cat("Total features created  :", ncol(model_features) - 1, "\n")
cat("Total rows for modeling :", nrow(model_features), "\n")
cat("Target variable         : team1_wins (Team1_Wins / Team2_Wins)\n")
cat("\nFeatures used:\n")
cat("  1. team1_encoded       - Team 1 identity\n")
cat("  2. team2_encoded       - Team 2 identity\n")
cat("  3. team1_won_toss      - Did team1 win the toss?\n")
cat("  4. toss_field          - Did toss winner choose to field?\n")
cat("  5. toss_decision_encoded - Field(1) or Bat(0)\n")
cat("  6. is_playoff          - Is it a knockout match?\n")
cat("  7. is_super_over       - Was there a super over?\n")
cat("  8. team1_win_rate      - Team1 historical win rate\n")
cat("  9. team2_win_rate      - Team2 historical win rate\n")
cat(" 10. win_rate_diff       - Difference in win rates\n")
cat(" 11. venue_match_count   - How big/important is the venue\n")
cat("==========================================\n")


# ============================================
# STEP 6: MATCH OUTCOME PREDICTION
# ============================================

# Load the feature dataset we built in Step 5
model_features <- read_csv("data/model_features.csv")

# Convert target back to factor (read_csv removes factor type)
model_features <- model_features %>%
  mutate(team1_wins = factor(team1_wins,
                             levels = c("Team2_Wins", "Team1_Wins")))

cat("Model features loaded!\n")
cat("Shape:", nrow(model_features), "rows,", ncol(model_features), "cols\n")
cat("Target distribution:\n")
print(table(model_features$team1_wins))


# Set seed for reproducibility (so results are same every time)
set.seed(42)

# 80% training, 20% testing
train_index <- createDataPartition(model_features$team1_wins,
                                   p = 0.80,
                                   list = FALSE)

train_data <- model_features[ train_index, ]
test_data  <- model_features[-train_index, ]

cat("\nData split complete!\n")
cat("Training rows :", nrow(train_data), "\n")
cat("Testing rows  :", nrow(test_data), "\n")

# Verify target balance is maintained in both splits
cat("\nTraining target distribution:\n")
print(round(prop.table(table(train_data$team1_wins)) * 100, 1))

cat("\nTesting target distribution:\n")
print(round(prop.table(table(test_data$team1_wins)) * 100, 1))


# Use 10-fold cross validation for reliable accuracy estimates
train_control <- trainControl(
  method          = "cv",        # cross validation
  number          = 10,          # 10 folds
  savePredictions = TRUE,
  classProbs      = TRUE,        # needed for ROC curve later
  summaryFunction = twoClassSummary
)

cat("Cross validation control set — 10-fold CV ready!\n")


cat("\n==========================================\n")
cat("  MODEL 1: LOGISTIC REGRESSION\n")
cat("==========================================\n")

set.seed(42)
model_lr <- train(
  team1_wins ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = train_control,
  metric    = "ROC"
)

# Predict on test data
pred_lr <- predict(model_lr, newdata = test_data)

# Confusion Matrix
cm_lr <- confusionMatrix(pred_lr, test_data$team1_wins, positive = "Team1_Wins")
print(cm_lr)

# Store accuracy
acc_lr <- round(cm_lr$overall["Accuracy"] * 100, 2)
cat("\nLogistic Regression Accuracy:", acc_lr, "%\n")


cat("\n==========================================\n")
cat("  MODEL 2: DECISION TREE\n")
cat("==========================================\n")

set.seed(42)
model_dt <- train(
  team1_wins ~ .,
  data      = train_data,
  method    = "rpart",
  trControl = train_control,
  metric    = "ROC",
  tuneLength = 10
)

# Predict on test data
pred_dt <- predict(model_dt, newdata = test_data)

# Confusion Matrix
cm_dt <- confusionMatrix(pred_dt, test_data$team1_wins, positive = "Team1_Wins")
print(cm_dt)

# Store accuracy
acc_dt <- round(cm_dt$overall["Accuracy"] * 100, 2)
cat("\nDecision Tree Accuracy:", acc_dt, "%\n")

# Visualize the Decision Tree
png("plots/10_decision_tree.png", width = 1200, height = 800, res = 100)
rpart.plot(model_dt$finalModel,
           type  = 4,
           extra = 104,
           box.palette = "GnBu",
           shadow.col  = "gray",
           nn    = TRUE,
           main  = "IPL Match Outcome - Decision Tree")
dev.off()
cat("Decision Tree plot saved!\n")


cat("\n==========================================\n")
cat("  MODEL 3: RANDOM FOREST\n")
cat("==========================================\n")

set.seed(42)
model_rf <- train(
  team1_wins ~ .,
  data      = train_data,
  method    = "rf",
  trControl = train_control,
  metric    = "ROC",
  tuneLength = 5,
  ntree      = 500
)

# Predict on test data
pred_rf <- predict(model_rf, newdata = test_data)

# Confusion Matrix
cm_rf <- confusionMatrix(pred_rf, test_data$team1_wins, positive = "Team1_Wins")
print(cm_rf)

# Store accuracy
acc_rf <- round(cm_rf$overall["Accuracy"] * 100, 2)
cat("\nRandom Forest Accuracy:", acc_rf, "%\n")


# Which features matter most for prediction?
importance_df <- varImp(model_rf)$importance %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

cat("\nFeature Importance (Random Forest):\n")
print(importance_df)

# Plot feature importance
ggplot(importance_df, aes(x = reorder(feature, Overall),
                          y = Overall, fill = Overall)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "#56B4E9", high = "#E69F00") +
  geom_text(aes(label = round(Overall, 1)), hjust = -0.1,
            size = 4, fontface = "bold") +
  labs(
    title = "Feature Importance — Random Forest Model",
    x     = "Feature",
    y     = "Importance Score"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  ylim(0, max(importance_df$Overall) * 1.15)

ggsave("plots/11_feature_importance.png", width = 10, height = 6)
cat("Feature importance chart saved!\n")


# Build comparison table
model_comparison <- data.frame(
  Model    = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(acc_lr, acc_dt, acc_rf),
  Sensitivity = c(
    round(cm_lr$byClass["Sensitivity"] * 100, 2),
    round(cm_dt$byClass["Sensitivity"] * 100, 2),
    round(cm_rf$byClass["Sensitivity"] * 100, 2)
  ),
  Specificity = c(
    round(cm_lr$byClass["Specificity"] * 100, 2),
    round(cm_dt$byClass["Specificity"] * 100, 2),
    round(cm_rf$byClass["Specificity"] * 100, 2)
  )
)

cat("\n==========================================\n")
cat("       MODEL COMPARISON SUMMARY\n")
cat("==========================================\n")
print(model_comparison)

# Plot model comparison
ggplot(model_comparison, aes(x = reorder(Model, Accuracy),
                             y = Accuracy, fill = Model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(Accuracy, "%")),
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#0072B2",
    "Decision Tree"       = "#E69F00",
    "Random Forest"       = "#009E73"
  )) +
  labs(
    title = "Model Accuracy Comparison — IPL Match Prediction",
    x     = "Model",
    y     = "Accuracy (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  ylim(0, 100)

ggsave("plots/12_model_comparison.png", width = 8, height = 6)
cat("Model comparison chart saved!\n")


# Get predicted probabilities from Random Forest
pred_rf_prob <- predict(model_rf, newdata = test_data, type = "prob")

# Compute ROC curve
roc_rf <- roc(
  response  = test_data$team1_wins,
  predictor = pred_rf_prob[, "Team1_Wins"],
  levels    = c("Team2_Wins", "Team1_Wins")
)

# Plot ROC Curve
png("plots/13_roc_curve.png", width = 800, height = 700, res = 100)
plot(roc_rf,
     col       = "#009E73",
     lwd       = 2.5,
     main      = paste("ROC Curve — Random Forest\nAUC =",
                       round(auc(roc_rf), 3)),
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "#009E7330")
dev.off()

cat("ROC curve saved!\n")
cat("AUC Score:", round(auc(roc_rf), 3), "\n")


cat("\n==========================================\n")
cat("     FINAL PREDICTION RESULTS\n")
cat("==========================================\n")
cat("Total matches tested     :", nrow(test_data), "\n")
cat("Logistic Regression      :", acc_lr, "%\n")
cat("Decision Tree            :", acc_dt, "%\n")
cat("Random Forest            :", acc_rf, "%\n")
cat("Best Model               :", 
    model_comparison$Model[which.max(model_comparison$Accuracy)], "\n")
cat("AUC Score (Random Forest):", round(auc(roc_rf), 3), "\n")
cat("\nNote: ~60-65% accuracy is excellent for cricket\n")
cat("prediction due to the sport's inherent unpredictability!\n")
cat("==========================================\n")


# Load pROC library and re-run ROC curve
library(pROC)

# Compute ROC curve
roc_rf <- roc(
  response  = test_data$team1_wins,
  predictor = pred_rf_prob[, "Team1_Wins"],
  levels    = c("Team2_Wins", "Team1_Wins")
)

# Plot and save ROC Curve
png("plots/13_roc_curve.png", width = 800, height = 700, res = 100)
plot(roc_rf,
     col             = "#009E73",
     lwd             = 2.5,
     main            = paste("ROC Curve — Random Forest\nAUC =",
                             round(auc(roc_rf), 3)),
     print.auc       = TRUE,
     auc.polygon     = TRUE,
     auc.polygon.col = "#009E7330")
dev.off()

cat("ROC curve saved!\n")
cat("AUC Score:", round(auc(roc_rf), 3), "\n")


# ============================================
# STEP 7: MODEL EVALUATION & PREDICTION
# ============================================

# Detailed metrics for our best model — Decision Tree
cat("==========================================\n")
cat("   DETAILED EVALUATION — DECISION TREE\n")
cat("==========================================\n")

# Extract all metrics from confusion matrix
dt_metrics <- cm_dt$byClass

cat("\nAccuracy     :", round(cm_dt$overall["Accuracy"] * 100, 2), "%\n")
cat("Sensitivity  :", round(dt_metrics["Sensitivity"] * 100, 2), "%\n")
cat("Specificity  :", round(dt_metrics["Specificity"] * 100, 2), "%\n")
cat("Precision    :", round(dt_metrics["Pos Pred Value"] * 100, 2), "%\n")
cat("F1 Score     :", round(dt_metrics["F1"] * 100, 2), "%\n")
cat("Balanced Acc :", round(dt_metrics["Balanced Accuracy"] * 100, 2), "%\n")


cat("\n==========================================\n")
cat("        WHAT EACH METRIC MEANS\n")
cat("==========================================\n")
cat("Accuracy     : Overall correct predictions out of all matches\n")
cat("Sensitivity  : When Team1 actually wins, how often did we predict it?\n")
cat("Specificity  : When Team2 actually wins, how often did we predict it?\n")
cat("Precision    : When we predicted Team1 wins, how often were we right?\n")
cat("F1 Score     : Balance between Sensitivity and Precision\n")
cat("Balanced Acc : Average of Sensitivity and Specificity\n")
cat("==========================================\n")


# See how model performed across each of the 10 folds
cat("\n--- Cross Validation Results (Decision Tree) ---\n")
cv_results <- model_dt$resample
print(cv_results)

cat("\nCV Summary:\n")
cat("Mean ROC  :", round(mean(cv_results$ROC), 3), "\n")
cat("SD ROC    :", round(sd(cv_results$ROC), 3), "\n")
cat("Mean Sens :", round(mean(cv_results$Sens), 3), "\n")
cat("Mean Spec :", round(mean(cv_results$Spec), 3), "\n")


# Plot ROC across 10 folds to check consistency
ggplot(cv_results, aes(x = 1:nrow(cv_results), y = ROC)) +
  geom_line(color = "#0072B2", linewidth = 1.2) +
  geom_point(color = "#E69F00", size = 4) +
  geom_hline(yintercept = mean(cv_results$ROC),
             linetype = "dashed", color = "#D55E00", linewidth = 1) +
  geom_text(aes(label = round(ROC, 3)), vjust = -1, size = 3.5) +
  labs(
    title    = "Decision Tree — ROC Across 10 CV Folds",
    subtitle = paste("Mean ROC =", round(mean(cv_results$ROC), 3)),
    x        = "Fold Number",
    y        = "ROC Score"
  ) +
  theme_minimal(base_size = 13) +
  ylim(min(cv_results$ROC) - 0.05, max(cv_results$ROC) + 0.05)

ggsave("plots/14_cv_stability.png", width = 10, height = 6)
cat("CV stability chart saved!\n")


# Display the decision tree with full detail
png("plots/15_decision_tree_full.png", width = 1400, height = 900, res = 100)
rpart.plot(
  model_dt$finalModel,
  type        = 4,
  extra       = 108,
  box.palette = "RdYlGn",
  shadow.col  = "gray80",
  nn          = TRUE,
  fallen.leaves = TRUE,
  main        = "IPL Match Outcome Prediction — Decision Tree",
  cex         = 0.85
)
dev.off()
cat("Full decision tree saved!\n")


# ============================================
# LIVE IPL MATCH PREDICTOR FUNCTION
# ============================================

predict_ipl_match <- function(team1_name, team2_name, 
                              venue_name    = "Eden Gardens",
                              toss_winner   = NULL,
                              toss_decision = "field",
                              is_playoff    = 0) {
  
  # List of valid teams
  valid_teams <- sort(unique(c(matches$team1, matches$team2)))
  
  # Validate team names
  if (!team1_name %in% valid_teams) {
    cat("❌ Invalid team1 name. Valid teams are:\n")
    print(valid_teams)
    return(NULL)
  }
  if (!team2_name %in% valid_teams) {
    cat("❌ Invalid team2 name. Valid teams are:\n")
    print(valid_teams)
    return(NULL)
  }
  if (team1_name == team2_name) {
    cat("❌ Team1 and Team2 cannot be the same team!\n")
    return(NULL)
  }
  
  # If no toss winner specified, assume team1 won toss
  if (is.null(toss_winner)) toss_winner <- team1_name
  
  # Get venue match count
  venue_count <- venue_counts %>%
    filter(venue == venue_name) %>%
    pull(venue_match_count)
  
  # If venue not found, use median
  if (length(venue_count) == 0) {
    venue_count <- median(venue_counts$venue_match_count)
    cat("ℹ️  Venue not found — using median venue size\n")
  }
  
  # Build input feature row
  new_match <- data.frame(
    team1_encoded         = team_encoding[team1_name],
    team2_encoded         = team_encoding[team2_name],
    team1_won_toss        = ifelse(toss_winner == team1_name, 1, 0),
    toss_field            = ifelse(toss_decision == "field", 1, 0),
    toss_decision_encoded = ifelse(toss_decision == "field", 1, 0),
    is_playoff            = is_playoff,
    is_super_over         = 0,
    team1_win_rate        = team_win_rate$win_rate[
      team_win_rate$team == team1_name],
    team2_win_rate        = team_win_rate$win_rate[
      team_win_rate$team == team2_name],
    win_rate_diff         = team_win_rate$win_rate[
      team_win_rate$team == team1_name] -
      team_win_rate$win_rate[
        team_win_rate$team == team2_name],
    venue_match_count     = venue_count
  )
  
  # Get prediction and probabilities
  prediction <- predict(model_dt, newdata = new_match)
  prob       <- predict(model_dt, newdata = new_match, type = "prob")
  
  # Display result
  cat("\n╔══════════════════════════════════════════╗\n")
  cat("║         IPL MATCH PREDICTION             ║\n")
  cat("╠══════════════════════════════════════════╣\n")
  cat("║ Team 1 :", formatC(team1_name, width = 32), "║\n")
  cat("║ Team 2 :", formatC(team2_name, width = 32), "║\n")
  cat("║ Venue  :", formatC(venue_name, width = 32), "║\n")
  cat("╠══════════════════════════════════════════╣\n")
  cat("║ Team 1 Win Probability :",
      formatC(paste0(round(prob$Team1_Wins * 100, 1), "%"), width = 17), "║\n")
  cat("║ Team 2 Win Probability :",
      formatC(paste0(round(prob$Team2_Wins * 100, 1), "%"), width = 17), "║\n")
  cat("╠══════════════════════════════════════════╣\n")
  
  winner <- ifelse(prediction == "Team1_Wins", team1_name, team2_name)
  cat("║ 🏆 PREDICTED WINNER :",
      formatC(winner, width = 20), "║\n")
  cat("╚══════════════════════════════════════════╝\n")
  
  return(invisible(list(winner = winner, probabilities = prob)))
}

cat("✅ Prediction function ready!\n")


# ============================================
# TEST THE PREDICTOR — TRY THESE MATCHES!
# ============================================

# Match 1: The biggest IPL rivalry
predict_ipl_match(
  team1_name    = "Mumbai Indians",
  team2_name    = "Chennai Super Kings",
  venue_name    = "Wankhede Stadium",
  toss_winner   = "Mumbai Indians",
  toss_decision = "field"
)

# Match 2: New vs Old — GT vs CSK
predict_ipl_match(
  team1_name    = "Gujarat Titans",
  team2_name    = "Chennai Super Kings",
  venue_name    = "Narendra Modi Stadium",
  toss_winner   = "Gujarat Titans",
  toss_decision = "field"
)

# Match 3: A playoff match — KKR vs RCB
predict_ipl_match(
  team1_name    = "Kolkata Knight Riders",
  team2_name    = "Royal Challengers Bengaluru",
  venue_name    = "Eden Gardens",
  toss_winner   = "Kolkata Knight Riders",
  toss_decision = "field",
  is_playoff    = 1
)

# Match 4: Bottom vs Top — Pune Warriors vs Mumbai Indians
predict_ipl_match(
  team1_name    = "Pune Warriors",
  team2_name    = "Mumbai Indians",
  venue_name    = "Wankhede Stadium",
  toss_winner   = "Mumbai Indians",
  toss_decision = "field"
)


# ============================================
# YOUR TURN — PREDICT ANY MATCH YOU WANT!
# ============================================

# Available teams to choose from:
cat("\nAvailable IPL Teams:\n")
print(sort(unique(c(matches$team1, matches$team2))))

# Available venues to choose from:
cat("\nTop 10 Venues:\n")
print(top_venues$venue)

# Now try your own match — change the names below!
predict_ipl_match(
  team1_name    = "Rajasthan Royals",       # ← Change this
  team2_name    = "Sunrisers Hyderabad",    # ← Change this
  venue_name    = "Eden Gardens",           # ← Change this
  toss_winner   = "Rajasthan Royals",       # ← Change this
  toss_decision = "field",                  # "field" or "bat"
  is_playoff    = 0                         # 1 if knockout, 0 if league
)


cat("\n╔══════════════════════════════════════════╗\n")
cat("║       COMPLETE PROJECT SUMMARY           ║\n")
cat("╠══════════════════════════════════════════╣\n")
cat("║ Dataset    : IPL 2008–2024               ║\n")
cat("║ Matches    : 1,090                       ║\n")
cat("║ Features   : 11 engineered variables     ║\n")
cat("║ Models     : 3 (LR, DT, RF)             ║\n")
cat("╠══════════════════════════════════════════╣\n")
cat("║ RESULTS:                                 ║\n")
cat("║ Logistic Regression  : 53.67%            ║\n")
cat("║ Decision Tree        : 55.50% ← Best    ║\n")
cat("║ Random Forest        : 51.83%            ║\n")
cat("║ AUC Score (RF)       : 0.533             ║\n")
cat("╠══════════════════════════════════════════╣\n")
cat("║ TOP FEATURES:                            ║\n")
cat("║ 1. Win Rate Difference  (Score: 100)     ║\n")
cat("║ 2. Venue Match Count    (Score:  99)     ║\n")
cat("║ 3. Team Win Rates       (Score:  52)     ║\n")
cat("╚══════════════════════════════════════════╝\n")
