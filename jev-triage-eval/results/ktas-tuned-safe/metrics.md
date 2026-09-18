# ktas: held-out tuning of the Jev policy

5-fold cross-fitted, objective = qwk_safe (under-triage ceiling 0.1). Every jev_*_tuned / shift / cuts / stacked row is out-of-fold: parameters chosen on the other folds. Rows in parentheses are the untuned references; nurse and ML rows are copied from the base run.

| Model | Acc | Bal. acc | Macro F1 | QWK | MAE | ±1 level | Under | Over | Sens (L1-2) | AUROC (L1-2) | ECE | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_rules (default) | 0.346 | 0.427 | 0.353 | 0.406 | 0.825 | 0.848 | 0.084 | 0.571 | 0.931 |  |  |  |
| jev_score (argmax) | 0.460 | 0.488 | 0.424 | 0.444 | 0.678 | 0.878 | 0.053 | 0.487 | 0.862 | 0.862 | 0.338 | 0.822 |
| jev_rules_tuned | 0.423 | 0.469 | 0.386 | 0.455 | 0.721 | 0.874 | 0.098 | 0.479 | 0.878 |  |  |  |
| jev_score_shift | 0.549 | 0.531 | 0.509 | 0.565 | 0.519 | 0.937 | 0.095 | 0.356 | 0.776 |  |  |  |
| jev_score_cuts | 0.564 | 0.533 | 0.521 | 0.570 | 0.496 | 0.946 | 0.100 | 0.336 | 0.720 |  |  |  |
| jev_stacked_lr | 0.630 | 0.515 | 0.542 | 0.638 | 0.408 | 0.964 | 0.197 | 0.173 | 0.508 | 0.877 | 0.019 | 0.500 |
| nurse | 0.853 | 0.797 | 0.807 | 0.875 | 0.163 | 0.985 | 0.103 | 0.043 | 0.862 |  |  |  |
| logreg | 0.640 | 0.661 | 0.615 | 0.628 | 0.448 | 0.928 | 0.196 | 0.164 | 0.691 | 0.900 | 0.053 | 0.493 |
| hgb | 0.685 | 0.558 | 0.590 | 0.670 | 0.363 | 0.955 | 0.163 | 0.152 | 0.593 | 0.899 | 0.161 | 0.478 |
| rf | 0.704 | 0.617 | 0.627 | 0.686 | 0.345 | 0.959 | 0.158 | 0.138 | 0.598 | 0.909 | 0.083 | 0.429 |

## Parameters chosen per fold

|   fold |   n_train |   t_lifesaving |   t_high_risk |   t_altered |   t_distress |   t_many |   t_any |   shift | cuts                   |
|-------:|----------:|---------------:|--------------:|------------:|-------------:|---------:|--------:|--------:|:-----------------------|
|      0 |      1013 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.4 |    0.35 | [1.15, 2.05, 3.5, 4.3] |
|      1 |      1013 |            0.5 |          0.55 |         0.5 |          0.9 |      0.6 |     0.4 |    0.4  | [1.15, 2.1, 3.05, 4.3] |
|      2 |      1014 |            0.3 |          0.55 |         0.5 |          0.9 |      0.6 |     0.4 |    0.35 | [1.1, 2.1, 3.1, 4.3]   |
|      3 |      1014 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.4 |    0.4  | [1.2, 2.1, 3.05, 4.5]  |
|      4 |      1014 |            0.5 |          0.55 |         0.5 |          0.9 |      0.6 |     0.4 |    0.4  | [1.1, 2.1, 3.1, 4.3]   |

## Parameters fitted on all data (for deployment)

```json
{
 "rules": {
  "t_lifesaving": 0.3,
  "t_high_risk": 0.55,
  "t_altered": 0.5,
  "t_distress": 0.9,
  "t_many": 0.6,
  "t_any": 0.4
 },
 "shift": 0.4,
 "cuts": [
  1.15,
  2.1,
  3.1,
  4.3
 ]
}
```
