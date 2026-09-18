# ktas (all, esi-v3): held-out tuning of the Jev policy

5-fold cross-fitted, objective = qwk_safe (under-triage ceiling 0.1). Every jev_*_tuned / shift / cuts / stacked row is out-of-fold: parameters chosen on the other folds. Rows in parentheses are the untuned references; nurse and ML rows are copied from the base run.

| Model | Acc | Bal. acc | Macro F1 | QWK | MAE | ±1 level | Under | Over | Sens (L1-2) | AUROC (L1-2) | ECE | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_rules (default) | 0.393 | 0.446 | 0.386 | 0.420 | 0.766 | 0.857 | 0.091 | 0.516 | 0.882 |  |  |  |
| jev_score (argmax) | 0.463 | 0.490 | 0.425 | 0.437 | 0.680 | 0.876 | 0.052 | 0.485 | 0.866 | 0.861 | 0.336 | 0.823 |
| jev_rules_tuned | 0.443 | 0.482 | 0.397 | 0.464 | 0.681 | 0.888 | 0.099 | 0.459 | 0.809 |  |  |  |
| jev_score_shift | 0.551 | 0.532 | 0.510 | 0.557 | 0.522 | 0.935 | 0.093 | 0.356 | 0.776 |  |  |  |
| jev_score_cuts | 0.537 | 0.508 | 0.488 | 0.548 | 0.519 | 0.949 | 0.103 | 0.360 | 0.659 |  |  |  |
| jev_stacked_lr | 0.627 | 0.517 | 0.541 | 0.637 | 0.413 | 0.962 | 0.199 | 0.174 | 0.528 | 0.876 | 0.032 | 0.502 |
| nurse | 0.853 | 0.797 | 0.807 | 0.875 | 0.163 | 0.985 | 0.103 | 0.043 | 0.862 |  |  |  |
| logreg | 0.640 | 0.661 | 0.615 | 0.628 | 0.448 | 0.928 | 0.196 | 0.164 | 0.691 | 0.900 | 0.053 | 0.493 |
| hgb | 0.685 | 0.558 | 0.590 | 0.670 | 0.363 | 0.955 | 0.163 | 0.152 | 0.593 | 0.899 | 0.161 | 0.478 |
| rf | 0.704 | 0.617 | 0.627 | 0.686 | 0.345 | 0.959 | 0.158 | 0.138 | 0.598 | 0.909 | 0.083 | 0.429 |

## Parameters chosen per fold

|   fold |   n_train |   t_lifesaving |   t_high_risk |   t_altered |   t_distress |   t_many |   t_any |   shift | cuts                    |
|-------:|----------:|---------------:|--------------:|------------:|-------------:|---------:|--------:|--------:|:------------------------|
|      0 |      1013 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.4 |    0.35 | [1.2, 2.05, 3.55, 4.4]  |
|      1 |      1013 |            0.3 |          0.8  |         0.3 |          0.9 |      0.5 |     0.4 |    0.4  | [1.2, 2.05, 3.3, 4.3]   |
|      2 |      1014 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.5 |    0.35 | [1.2, 2.1, 3.15, 4.4]   |
|      3 |      1014 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.5 |    0.4  | [1.3, 2.05, 3.25, 4.55] |
|      4 |      1014 |            0.3 |          0.75 |         0.5 |          0.9 |      0.5 |     0.5 |    0.35 | [1.2, 2.1, 3.1, 4.4]    |

## Parameters fitted on all data (for deployment)

```json
{
 "rules": {
  "t_lifesaving": 0.3,
  "t_high_risk": 0.75,
  "t_altered": 0.5,
  "t_distress": 0.9,
  "t_many": 0.5,
  "t_any": 0.5
 },
 "shift": 0.35,
 "cuts": [
  1.2,
  2.05,
  3.4,
  4.55
 ]
}
```
