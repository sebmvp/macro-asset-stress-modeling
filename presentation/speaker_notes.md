# Speaker notes — Macro Asset Stress Modeling

About eight minutes. Results, not a methods class. I say "we" for the
group work and "I" for the modeling comparison I actually owned.

If a number is in these notes, it is a **published report number**.

---

## Slide 1 — Problem

Hello — I'm Sebastian. This is a CompSci 390B group project on Gold, SPY, and Bitcoin under macro stress.

The question is simple: when markets get stressed, does Gold actually protect capital, and does Bitcoin behave like "digital Gold"?

A safe haven isn't an asset with a nice average return. It's an asset that defends *exactly when* equities are under pressure. That's the test we ran.

I worked on model design and evaluation. Dhruv built most of the empirical pipeline. Nathan led the written report. I'll walk the results.

## Slide 2 — Data and regimes

We used 4,150 daily observations, October 2014 through February 2026, from a public Kaggle panel. Gold, SPY, Bitcoin, plus VIX, high-yield spreads, and a financial stress index.

Log returns, because prices weren't stationary.

Stress is the 66th percentile of three proxies. That keeps elevated-stress days without emptying the sample. About a third of days trip each flag. `stress_any` is 56% of days; `stress_all` is 14%.

We re-ran the tests at the 75th and 90th percentiles later. If a finding only exists at one cutoff, we don't trust it.

## Slide 3 — Why the project changed direction

Milestone 1 tried to predict next-day Gold returns. Ridge could not beat a zero baseline. Volatility was a different story — persistence and Ridge already had R² in the 0.87–0.90 range on five-day SPY vol.

So we stopped asking "can we predict Gold tomorrow?" and started asking "when does Gold actually behave defensively, and under what kind of stress?"

That turn is the project.

## Slide 4 — Gold as a conditional safe haven

Under VIX stress, Gold's mean daily log return is +0.00068. SPY is −0.00092. The Gold–SPY spread widens by 0.00247, and the bootstrap interval excludes zero.

Composite stress looks the same direction, weaker. Financial-stress-index stress does **not** show a Gold advantage.

Takeaway: Gold is a conditional safe haven, not a universal one. Volatility-style stress is where the defensive behavior shows up.

## Slide 5 — Volatility, correlation, Bitcoin

When `stress_any` is on, Gold vol rises 25%. SPY vol rises 78%. Bitcoin sits near 61% in *both* regimes. Unchanged Bitcoin vol is not defensiveness — it's "always violent."

Gold's beta to SPY falls in stress, 0.16 to 0.02. Bitcoin's beta *rises*, 0.67 to 0.86. Bitcoin shares more of SPY's worst 5% days than Gold does.

I would not call Bitcoin digital Gold on this sample.

## Slide 6 — Three event windows

COVID 2020 is the textbook case. Gold +6.4%, SPY −14.5%, Bitcoin −17.5%. Gold's max drawdown is about a third of SPY's.

2022 inflation and rate hikes: Gold −11.9%, SPY −5.9%. Gold does not outperform. That's the counterexample that makes the conclusion conditional.

2023 banking stress: Gold beats SPY, but Bitcoin rallies more.

Different shocks, different "safe" assets. Don't average them into one slogan.

## Slide 7 — Models E, F, G

Three tasks, chronological splits, no shuffling.

Model E: will Gold beat SPY tomorrow? XGBoost ROC–AUC 0.6835 versus 0.50 majority and 0.50 VIX rule. Random Forest actually wins balanced accuracy. I care about that distinction — ranking winner and classification winner are not the same model.

Model F: forecast sticky volatility. Ridge and persistence beat XGBoost. XGBoost's five-day VIX R² is negative. Complexity is not free.

Model G: stress five days ahead. The original high-yield target was 0.7% positive on the test set, so accuracy was useless. On `stress_any`, logistic PR–AUC 0.7095 beats XGBoost.

I helped design and evaluate that comparison. The lesson I took: match the model class to the target.

## Slide 8 — Robustness

Walk-forward, expanding window, 2019 through 2026: mean ROC–AUC 0.6571, every fold above 0.55. Not one lucky split.

SHAP on Model E: 40% of the mass is cross-asset momentum, 20% macro stress. It is a momentum-plus-stress classifier, not a pure stress gauge.

VIX widening survives 66 / 75 / 90. `stress_any` dies at 90. HYS and FSI never were robust.

## Slide 9 — Close

Three findings. Stress tracks volatility. Gold is a conditional safe haven. Bitcoin is not digital Gold here.

Limitations I want on the table: full-sample quantiles, regime drift — the MLP's 47-point train/test gap is the exhibit — event-window choice, 2014 start, no trading costs, and the original notebooks were not recovered. The report is the source of truth for the numbers.

Happy to take questions on the modeling comparison or on why we moved off daily Gold-return prediction.
