# Evidential reasoning for life-insurance underwriting: how to run

## Setup
```
pip install -r requirements.txt
```
Put Kaggle's `train.csv` (Prudential Life Insurance Assessment) in `data/`. Do not commit the data: `data/` is listed in `.gitignore`.

## 1. Run the tests first
```
python -m pytest -q
```
Nine tests check that the implementation is the ER rule:
- With every reliability set to 1, it equals Dempster's rule (checked against an independent set-based implementation).
- With likelihood evidence, w = r = 1 and the class prior as one piece of evidence, it equals Bayes' rule.
- Zero weight and zero reliability means the evidence has no effect.
- The order of combination does not change the result.
- A missing value gives exactly the same result as removing that evidence.
- The fixed-parameter model equals binned naive Bayes.
- Learning the weights and reliabilities improves the validation likelihood.

## 2. Quick check (about 1–3 minutes)
```
python run_experiment.py --data data/train.csv --out results_quick --quick
```

## 3. Full run
```
python run_experiment.py --data data/train.csv --out results
```
On a laptop CPU this takes roughly 15–40 minutes, most of it for the 1,000-resample bootstrap. Use `--n-boot 200` for a faster draft.

## Output files (`results/`)
| File | What it answers |
|---|---|
| `test_metrics_summary.csv` | RQ1: every model on the test set. Metrics: QWK (argmax and expected rating), accuracy, log loss, Brier, ECE, macro AUROC, each with a 95% bootstrap CI |
| `paired_tests_er_vs_others.csv` | McNemar test and paired-bootstrap QWK difference, ER vs each model |
| `missing_data_stress_test.csv` | RQ2: all metrics at 0/10/30/50% extra MCAR missingness |
| `referral_tradeoff.csv` | RQ4: referral rate vs error on auto-decided cases |
| `underwriter_explanations.md` | RQ3: evidence-group effects for three test applicants |
| `er_learned_weights_reliabilities.csv`, `er_parameters_by_group.csv` | learned weight and reliability of every piece of evidence |
| `reliability_tables.csv` | calibration data for reliability diagrams |
| `run_info.json` | seed, split sizes, versions, runtime |

## Conventions
- Labels are 0..7 everywhere in the code (`Response - 1`, done once in `data.py`). Add 1 only when reporting ratings on the original 1–8 scale.
- Brier is the multi-class version (range 0–2).
- Rating "1" is not assumed to be the lowest or the highest risk; check the competition data page before interpreting the direction.

## Limitations to state in any write-up
- Each variable is one piece of evidence, so inputs are treated as conditionally independent. MAKER's joint evidence for interdependent inputs is not implemented.
- Weights and reliabilities are fitted by likelihood alone, not MAKER's bi-objective optimisation.
- Bins are fixed (quantiles or categories); referential values are not optimised.
- The features are anonymised, so explanations are at feature-group level.
