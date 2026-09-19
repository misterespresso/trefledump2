# Can a chain of Jev calls do what one call cannot?

Jev scores at chance on the weekday of a date. The question is whether the task can be broken into steps small enough for it, and the answer recomposed. Three sub-skills are measured separately, then combined into a working pipeline.

| Step | What is asked | Correct | Rate | 95% CI | Mean top probability |
|---|---|---|---|---|---|
| Arithmetic | `N mod 7` for N of 0 to 30, over options 0-6 | 186 / 186 | 100.0% | 97.98%-100.00% | 0.932 |
| Arithmetic, larger | `N mod 7` for N of 31 to 120 | 71 / 90 | 78.9% | 69.37%-86.05% | 0.654 |
| Weekday offset | K days after a named weekday, K of 0 to 6 | 196 / 196 | 100.0% | 98.08%-100.00% | 0.979 |
| Weekday offset, larger | K days after a named weekday, K of 7 to 30 | 86 / 168 | 51.2% | 43.69%-58.64% | 0.539 |
| Fact recall | the weekday of 1 January of a given year | 11 / 219 | 5.0% | 2.83%-8.77% | 0.161 |

The two arithmetic-style steps are perfect inside the range a date needs, and degrade outside it. The step that fails is neither: it is recalling a fact.

## The chain, end to end

Three stages on the same 200 dates used elsewhere in this repo. Jev supplies the weekday of 1 January for that year; code computes the day-of-year offset modulo 7 and never asks the model for it; Jev applies that offset to whatever stage one returned. Stage three is built on stage one's answer even when that answer is wrong, so these are the chain's own numbers and not a hybrid with ground truth. Abstentions in stage one are counted as failures.

| | Stage 1 correct | Stage 1 abstained | End to end | 95% CI |
|---|---|---|---|---|
| Stage 1 from Jev's own memory | 11 / 200 (5.5%) | 143 | **11 / 200 = 5.5%** | 3.10%-9.58% |
| Stage 1 from a table in `state` | 200 / 200 (100.0%) | 0 | **200 / 200 = 100.0%** | 98.12%-100.00% |
| Asking the date outright, no chain | | | 19 / 200 = 9.5% | 6.17%-14.36% |

The retrieval-fed chain against asking outright: Fisher's exact, two-sided p = 2.1e-92.

## The composition itself is not the problem

Across both chains, stage three was handed a correct starting weekday 211 times and produced the right answer in 211 of them (100.0%). It was handed a wrong one 46 times and produced the right answer in 0 (0.0% ) of those.

Stage three is a faithful function of its input. The chain is worth exactly as much as the fact fed into it, which is the ordinary behaviour of a pipeline rather than a defect of the model.

## What this says about chaining

- **Composition works.** Given a correct input, the final step applied a code-computed offset essentially without error. Small typed judgments do chain.
- **It can do arithmetic, within a range.** `N mod 7` is perfect to N of 30 and falls to 79% from 31 to 120. Weekday offsets are perfect to 6 days and fall to 51% from 7 to 30. Keep each step inside the window and the chain holds; let a step run past it and the chain inherits the error.
- **Recall is the weak link, and it knows it.** Asked the weekday of 1 January for a year, it abstained on 143 of 200 and was right on 11. That abstention is the useful behaviour: the failure is visible to the caller rather than silent.
- **So the architecture is retrieval plus judgment plus code.** Supply the facts, keep each judgment small, and do the arithmetic outside the model. That is also what TypeSafe's own guidance says to do.
