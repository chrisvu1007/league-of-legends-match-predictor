# Predicting League of Legends Match Outcomes

By Christopher Vu

League of Legends (LoL) is a 5v5 strategy game made by Riot Games and one of the biggest esports in the world. Two teams, one on the **Blue** side of the map and one on **Red**, take turns banning and picking champions in a draft and then play a single match that ends with one winner. This project looks at professional LoL matches and asks a simple question: **can we predict which team wins using only the information we have before the game starts?**

The data comes from [Oracle's Elixir](https://oracleselixir.com/tools/downloads), the standard public source for pro LoL stats. The full dataset has almost 60,000 rows covering 4,986 matches from the Winter and Spring 2026 seasons. Each match is recorded as 12 rows: 10 player rows plus 2 team-summary rows.

## Introduction

### General Introduction

Most match analysis uses things that happen during the game, like kills and gold. I wanted to use only pre-game information: which side a team is on, whether it picked first, the patch (game version), the league, and the champions banned and picked. The thing I'm predicting is `result`, which is whether a team won.

This is timely because in the 2026 season Riot changed the rules so the team with priority can choose either first pick or its preferred side. That only matters if those things give an advantage, so before building a model I look at one motivating question: **does Blue side win more often than Red side?**

### Introduction of Columns

After reshaping the data (see the next section), these are the columns that matter most:

| Column | Description |
| --- | --- |
| `side` | Which side the team played, Blue or Red. |
| `firstPick` | Whether the team picked first in the draft. |
| `patch` | The game version. Balance changes between patches, which changes the "meta." |
| `league` | The league the match was played in (LCK, LPL, LEC, etc.). |
| `teamname` | The team. I use this later to estimate team strength. |
| `ban1`–`ban5` | The five champions the team banned. |
| `pick1`–`pick5` | The five champions the team picked. |
| `date`, `year`, `split` | When the match was played. Used to split the data into past and future. |
| `result` | The target: whether the team won. |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Each match is stored as 12 rows: 10 player rows and 2 team-summary rows (one per team). Since my question is about teams, I only need the team rows, so I kept the 2 team rows per match. This leaves 9,972 rows. I also dropped the columns I didn't need and fixed the data types so that dates are read as dates and numbers as numbers. The two team rows are labeled with participant IDs 100 (Blue) and 200 (Red), which makes them easy to pull out.

### Univariate Analysis

First I counted how many games come from each league. This is mostly a check to see whether the data leans heavily on a few regions.

<iframe src="assets/games_by_league.html" width="100%" height="500" style="border:0;"></iframe>

A few big regions make up most of the data, with a lot of smaller leagues in the tail. This matters later when we look at missing bans.

### Bivariate Analysis

Next I compared the win rate of Blue-side teams and Red-side teams. If side didn't matter, both bars would sit at 50%.

<iframe src="assets/winrate_by_side.html" width="100%" height="500" style="border:0;"></iframe>

Blue is a little above 50% and Red is a little below. It's a small gap, which is why I test it properly in the hypothesis section.

### Interesting Aggregates

Side and first pick are related, so I also looked at the win rate for each combination of side and whether the team picked first.

<iframe src="assets/side_firstpick_winrate.html" width="100%" height="500" style="border:0;"></iframe>

| Side | Picked first | Did not pick first |
| --- | --- | --- |
| Blue | ? | ? |
| Red | ? | ? |

*(Fill in these four cells with the values from your pivot table.)*

Reading across the rows shows whether the side advantage still holds once you account for who picked first.

## Assessment of Missingness

### NMAR Analysis

The columns with the most missing values are the bans (`ban1` to `ban5`).

A column is NMAR ("not missing at random") if a value is missing because of what the value would have been, and nothing else in the data explains it. You can't test for NMAR; you can only argue it. One NMAR story here is that bans go unrecorded in the kinds of games where the draft wasn't tracked carefully. The extra information that would turn this into a testable MAR situation would be something like a flag for whether the data source recorded the draft phase at all.

But we already have a column that explains the missing bans well: `league`. So I tested that instead.

### Missingness Dependency

I ran a permutation test. The idea: if missing bans had nothing to do with the league, then shuffling the "missing or not" labels should look like the real data. I shuffled 10,000 times and measured how different the league mix was each time, using total variation distance (TVD), which is just a measure of how far apart two distributions are.

**Test 1 — bans vs. league**

- Null: the distribution of `league` is the same whether or not a ban is missing.
- Alternative: the distribution of `league` is different.

<iframe src="assets/missingness_tvd.html" width="100%" height="500" style="border:0;"></iframe>

The observed value (dashed line) is far past everything the shuffles produced, so the p-value is basically 0. I reject the null. Some leagues just don't record bans reliably, so the missingness depends on `league` (MAR, not NMAR).

**Test 2 — bans vs. result**

- Null: the distribution of `result` is the same whether or not a ban is missing.
- Alternative: the distribution of `result` is different.

Here the observed value sits inside the range of the shuffles, so the p-value is above 0.05 and I fail to reject. Whether a team won doesn't tell us anything about whether its bans were recorded.

So the missingness of bans depends on `league` but not on `result`.

## Hypothesis Testing

The win-rate chart suggested Blue wins a bit more than half its games, but a small gap could be noise, so I tested it.

- Null: Blue side's true win rate is 0.50.
- Alternative: Blue side's win rate is not 0.50.
- Test statistic: Blue side's observed win rate.
- Significance level: 0.05.

I used a bootstrap (resampling the matches many times to see how much the win rate naturally varies) together with a standard proportion test.

<iframe src="assets/hypothesis_bootstrap.html" width="100%" height="500" style="border:0;"></iframe>

The whole distribution sits to the right of 0.50, and the p-value is below 0.05, so I reject the null. Blue side's win rate is small but really above 50%. So there is a real side advantage, even if it's weak.

## Framing a Prediction Problem

The main task is predicting the winner. This is a **binary classification** problem: for each team, the model predicts win or lose. The response variable is `result`.

The model can only use information available before the match: side, first pick, patch, league, the team, and the champions drafted. Anything from during the game (kills, gold, objectives) is not allowed, because at prediction time it hasn't happened yet.

I use **accuracy** as the metric. Accuracy works here because the classes are balanced: every match has exactly one winner and one loser among its two team rows, so the data is split 50/50. (If the classes were lopsided, accuracy could be misleading and something like F1 would be better, but that's not the case here.) Guessing should get about 50%, so anything well above that is real.

## Baseline Model

The baseline uses two features: `side` and `firstPick`. Both are nominal (categories with no order), so there are 0 quantitative features, 0 ordinal features, and 2 nominal features. I one-hot encoded both and used a gradient boosting classifier.

I split the data chronologically, training on the earliest 75% of matches and testing on the most recent 25%. This is closer to how you'd actually use the model (predicting future games from past ones) and avoids letting the model see the future.

The baseline gets about **52% accuracy**, which is barely better than guessing. That's expected: side and first pick are weak signals on their own. The obvious thing missing is how good the two teams are.

## Final Model

The final model adds two features that reflect how games are actually won.

**1. Team strength (Elo).** The biggest factor in who wins is which team is better. I built an Elo rating for each team, which is the same idea used to rank chess players: teams gain points for beating strong opponents and lose points for losing to weak ones, updated after every game. A team's rating going into a match only uses games that happened before it, so the model never sees the result it's trying to predict. I gave the major regions different starting ratings (LCK and LPL highest) so the ratings make sense from the start.

**2. Draft composition.** Which champions a team drafts also matters. I turned each team's picks into a "bag of champions" (a column for every champion, marking which ones were picked) and used truncated SVD to compress that into 5 numbers that capture the main patterns in how teams draft.

I used a random forest and tuned it with grid search and cross-validation, which tries different settings and keeps the best one. It picked a shallow tree (max depth of 2), which suggests the signal is real but small and that a deeper model would just overfit.

On the same held-out matches, the final model gets about **XX%** accuracy, a clear improvement over the 52% baseline. Most of the gain comes from the Elo feature. The takeaway is intuitive: the best pre-game predictor isn't which side you're on, it's which team is better, with the draft and side advantage adding smaller effects on top.

## Fairness Analysis

A model can be accurate overall but still work better for one group than another. Since side is central to this project, the fairness question is: **does the model predict Blue-side teams as accurately as Red-side teams?**

- Group X: teams on Blue side. Group Y: teams on Red side.
- Metric: accuracy.
- Null: the model is fair; its accuracy is the same for both sides.
- Alternative: the model is unfair; its accuracy is different between sides.
- Test statistic: the absolute difference in accuracy between the two groups.
- Significance level: 0.05.

I ran a permutation test, shuffling the side labels and measuring the accuracy gap each time.

<iframe src="assets/fairness_gap.html" width="100%" height="500" style="border:0;"></iframe>

The observed gap sits inside the range of the shuffles, and the p-value is about **0.78**, well above 0.05, so I fail to reject the null. The model predicts both sides about equally well, so it looks fair.

### Conclusion

Side does matter, but only a little: Blue wins a bit more than half its games. Pre-game information is enough to beat a coin flip, but most of that comes down to one thing, which is that the better team usually wins. The draft and side advantage add smaller effects on top, and the final model is fair across both sides.
