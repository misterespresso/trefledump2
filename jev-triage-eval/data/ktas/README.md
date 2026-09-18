# KTAS triage dataset (1,267 adult ED visits)

Source: Moon SH, Shim JL, Park KS, Park CS. *Triage accuracy and causes of
mistriage using the Korean Triage and Acuity Scale.* PLOS ONE 14(9): e0216972
(2019). https://doi.org/10.1371/journal.pone.0216972

Data deposited on figshare (CC BY 4.0):
https://figshare.com/articles/dataset/9779267

`ktas_triage.csv` is a byte-for-byte copy of the file as redistributed on
Kaggle and in several public GitHub repositories (semicolon separated,
latin-1, decimal comma in `KTAS duration_min`). Two Korean EDs (one local, one
regional), October 2016 to September 2017. `KTAS_expert` is the consensus
level of three triage experts and is the ground truth; `KTAS_RN` is the level
assigned by the triage nurse at the time of the visit.

Columns (from the paper's data dictionary):

| Column | Meaning |
| --- | --- |
| Group | 1 local ED, 2 regional ED |
| Sex | 1 female, 2 male |
| Age | years |
| Patients number per hour | ED census at the time of triage |
| Arrival mode | 1 walking, 2 public ambulance (119), 3 private car, 4 private ambulance, 5 public transport / police, 6 wheelchair, 7 other |
| Injury | 1 non-injury, 2 injury |
| Chief_complain | free text |
| Mental | 1 alert, 2 verbal response, 3 pain response, 4 unresponsive |
| Pain | 1 pain, 0 no pain (the paper says 2 for no pain; the file uses 0) |
| NRS_pain | numeric rating scale 0-10 |
| SBP, DBP, HR, RR, BT, Saturation | initial nursing vitals; BT in Celsius |
| KTAS_RN | nurse KTAS level 1-5 |
| Diagnosis in ED, Disposition, Error_group, Length of stay_min, KTAS duration_min, mistriage | outcomes, never used as model input |
| KTAS_expert | expert KTAS level 1-5 (target) |

Level 1 is resuscitation, level 5 non-urgent. The scale is derived from the
Canadian Triage and Acuity Scale and maps closely onto ESI levels.
