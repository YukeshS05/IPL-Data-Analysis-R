# 🏏 IPL Data Analysis & Match Outcome Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Sports%20Analytics-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-3%20ML%20Models-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20Accuracy-55.50%25-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seasons-2008--2024-red?style=for-the-badge"/>
</p>

---

## 📋 Project Details

| Field | Details |
|---|---|
| **Subject** | R for Data Science (ISWE209L) |
| **Institution** | VIT University, Vellore |
| **Department** | School of Computer Science Engineering and Information Systems (SCORE) |
| **Programme** | M.Tech Software Engineering |
| **Guide** | Dr. Ranichandra C, Associate Professor |

---

## 👨‍💻 Team Members

| Name | Register Number |
|---|---|
| Monishkumar G | 23MIS0230 |
| Mukeshkumar B | 23MIS0637 |
| Yukesh S | 23MIS0352 |

---

## 📌 Project Overview

This project performs a **complete end-to-end data science pipeline** on IPL (Indian Premier League) cricket data using the **R programming language**. It covers everything from raw data loading to machine learning-based match outcome prediction.

The IPL is the world's most popular T20 cricket league with 17 seasons (2008–2024), 1,090 matches, and 260,920 ball deliveries — making it a rich dataset for sports analytics.

---

## 🎯 Objectives

- 🔍 Explore and visualize IPL match data across **17 seasons (2008–2024)**
- 🧹 Systematically clean raw data by resolving **8 data quality issues**
- 📊 Generate **8 professional EDA visualizations** using ggplot2
- ⚙️ Engineer **11 meaningful predictive features** from match and ball data
- 🤖 Build and compare **3 machine learning classification models**
- 📈 Evaluate models using Accuracy, Sensitivity, Specificity, F1 Score, and AUC-ROC
- 🏆 Build a **live match prediction function** for any two IPL teams

---

## 📁 Repository Structure

```
IPL-Data-Analysis-R/
│
├── 📂 data/
│   ├── matches.csv                # Original match-level dataset
│   ├── matches_clean.csv          # Cleaned match dataset
│   └── model_features.csv         # Engineered features for ML models
│
├── 📂 plots/
│   ├── 01_total_wins_by_team.png
│   ├── 02_toss_decision_trend.png
│   ├── 03_toss_impact.png
│   ├── 04_batting_vs_chasing.png
│   ├── 05_top_batsmen.png
│   ├── 06_top_bowlers.png
│   ├── 07_matches_per_season.png
│   ├── 08_top_venues.png
│   ├── 09_correlation_heatmap.png
│   ├── 10_decision_tree.png
│   ├── 11_feature_importance.png
│   ├── 12_model_comparison.png
│   ├── 13_roc_curve.png
│   ├── 14_cv_stability.png
│   └── 15_decision_tree_full.png
│
├── 📂 scripts/
│   └── analysis.R                 # Complete R code — Steps 1 to 7
│
├── 📂 report/
│   └── IPL_Report.Rmd             # R Markdown report
│
├── .gitignore
└── README.md
```

---

## 🗂️ Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `matches.csv` | 1,095 | 20 | One row per match — season, teams, toss, venue, winner |
| `deliveries.csv` | 260,920 | 17 | One row per ball — batsman, bowler, runs, wickets |

**Source:** [IPL Complete Dataset — Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 🔧 R Packages Used

| Package | Purpose |
|---|---|
| `tidyverse` | Data manipulation and pipeline operations |
| `ggplot2` | Professional data visualizations |
| `dplyr` | Data filtering, grouping, summarization |
| `readr` | Fast CSV file reading |
| `caret` | ML model training with cross-validation |
| `rpart` + `rpart.plot` | Decision Tree model and visualization |
| `randomForest` | Random Forest ensemble model |
| `corrplot` | Correlation heatmap |
| `pROC` | ROC curve and AUC calculation |

---

## 🧹 Data Cleaning Steps

| # | Issue Found | Fix Applied |
|---|---|---|
| 1 | Season format mixed (2007/08, 2009/10) | Extracted first 4 digits as integer |
| 2 | 4 team name inconsistencies | Unified using `case_when()` |
| 3 | 5 matches with no winner | Removed with `filter(!is.na(winner))` |
| 4 | 51 missing city values | Filled from venue name using `str_detect()` |
| 5 | `method` column 98% empty | Dropped with `select(-method)` |
| 6 | `umpire1/umpire2` irrelevant | Dropped |
| 7 | Over numbering 0–19 | Corrected to 1–20 with `+1` |
| 8 | No toss impact column | Created `toss_winner_won` feature |

---

## 📊 Key EDA Findings

### 🏆 Top Teams (All-Time Wins)

| Rank | Team | Wins | Win Rate |
|---|---|---|---|
| 1 | Mumbai Indians | 144 | 55.2% |
| 2 | Chennai Super Kings | 138 | 58.2% |
| 3 | Kolkata Knight Riders | 131 | 52.2% |
| 4 | Gujarat Titans | 28 | 62.2% *(since 2022)* |

### 🏏 Top Run Scorers

| Rank | Player | Runs |
|---|---|---|
| 1 | Virat Kohli | 8,014 |
| 2 | Shikhar Dhawan | 6,769 |
| 3 | Rohit Sharma | 6,630 |

### 🎯 Top Wicket Takers

| Rank | Player | Wickets |
|---|---|---|
| 1 | YS Chahal | 205 |
| 2 | PP Chawla | 192 |
| 3 | DJ Bravo | 183 |

### 🎲 Toss Impact

> Toss winners win only **50.8%** of matches — barely above a coin flip.
> The toss has **negligible impact** on IPL match outcomes.

---

## ⚙️ Feature Engineering

11 features were engineered for the prediction model:

| # | Feature | Description |
|---|---|---|
| 1 | `team1_encoded` | Team 1 mapped to integer ID (1–15) |
| 2 | `team2_encoded` | Team 2 mapped to integer ID (1–15) |
| 3 | `team1_won_toss` | Did Team 1 win the toss? (0/1) |
| 4 | `toss_field` | Did toss winner choose to field? (0/1) |
| 5 | `toss_decision_encoded` | Field=1, Bat=0 |
| 6 | `is_playoff` | Knockout match flag (0/1) |
| 7 | `is_super_over` | Super over flag (0/1) |
| 8 | `team1_win_rate` | Team 1 historical win rate |
| 9 | `team2_win_rate` | Team 2 historical win rate |
| 10 | `win_rate_diff` | team1_win_rate − team2_win_rate |
| 11 | `venue_match_count` | Total IPL matches at this venue |

---

## 🤖 Machine Learning Results

### Model Comparison

| Model | Accuracy | Sensitivity | Specificity | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 53.67% | 56.76% | 50.47% | ~54.5% |
| **Decision Tree** | **55.50%** ✅ | **60.36%** | 50.47% | **58.01%** |
| Random Forest | 51.83% | 43.24% | 60.75% | ~47.0% |

> **Best Model: Decision Tree — 55.50% accuracy**

### Experimental Setup

- **Train/Test Split:** 80% Training (872 rows) / 20% Testing (218 rows)
- **Cross Validation:** 10-fold CV — Mean ROC = 0.567, SD = 0.061 *(stable)*
- **Random Seed:** `set.seed(42)` for reproducibility
- **AUC Score (Random Forest):** 0.533

### Feature Importance (Random Forest)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `win_rate_diff` | 100.0 |
| 2 | `venue_match_count` | 98.8 |
| 3 | `team2_win_rate` | 52.1 |
| 4 | `team1_win_rate` | 51.9 |
| 11 | `is_super_over` | 0.0 *(pure luck)* |

---

## 🏆 Live Match Prediction Results

| Match | Venue | Predicted Winner | Probability |
|---|---|---|---|
| MI vs CSK | Wankhede Stadium | Chennai Super Kings | CSK 58.6% |
| GT vs CSK | Narendra Modi Stadium | Gujarat Titans | GT 59.7% |
| KKR vs RCB *(Playoff)* | Eden Gardens | Kolkata Knight Riders | KKR 59.7% |
| Pune Warriors vs MI | Wankhede Stadium | Mumbai Indians | MI 58.6% |
| RR vs SRH | Eden Gardens | Rajasthan Royals | RR 59.7% |

---

## ▶️ How to Run

```r
# 1. Clone the repository
# git clone https://github.com/YukeshS05/IPL-Data-Analysis-R.git

# 2. Open RStudio and set working directory
setwd("path/to/IPL-Data-Analysis-R")

# 3. Install required packages
install.packages(c("tidyverse","ggplot2","dplyr","readr","caret",
                   "rpart","rpart.plot","randomForest","corrplot","pROC"))

# 4. Run the full analysis
source("scripts/analysis.R")

# 5. Predict any match
predict_ipl_match(
  team1_name    = "Mumbai Indians",
  team2_name    = "Chennai Super Kings",
  venue_name    = "Wankhede Stadium",
  toss_winner   = "Mumbai Indians",
  toss_decision = "field"
)
```

---

## 📌 Key Conclusions

- **Mumbai Indians** have the most wins (144) across 17 seasons
- **Gujarat Titans** have the highest win rate (62.2%) since their 2022 debut
- **Virat Kohli** leads all-time run scoring with 8,014 runs — 1,245 ahead of 2nd place
- **YS Chahal** leads all-time wickets with 205 — leg spin dominates T20
- **Toss impact is statistically negligible** — only 50.8% win rate for toss winners
- **Win rate difference** between teams is the single strongest match predictor
- **55.50% accuracy** is meaningful — cricket is inherently unpredictable with pre-match data only

---

## 🔮 Future Work

- Add ball-by-ball **live match features** (powerplay score, wickets) to target 70–80% accuracy
- Include **player-level features**: recent form, batting average, bowling economy
- Apply **LSTM deep learning** for sequence-based over-by-over prediction
- Build a **Shiny web app** for interactive real-time match prediction
- Use **ensemble stacking** to combine all 3 model predictions

---

## 📚 References

1. [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
2. Wickham et al. (2019) — Welcome to the tidyverse. *JOSS*, 4(43)
3. Kuhn, M. (2008) — Building Predictive Models in R Using caret. *JSS*, 28(5)
4. Breiman, L. (2001) — Random Forests. *Machine Learning*, 45(1)
5. Robin et al. (2011) — pROC: ROC curves for R. *BMC Bioinformatics*, 12

---

<p align="center">
  Made with ❤️ using R &nbsp;|&nbsp; VIT University, Vellore &nbsp;|&nbsp; 2025
</p>
