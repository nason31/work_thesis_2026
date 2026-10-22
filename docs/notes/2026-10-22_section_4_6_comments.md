# Comments on Section 4.6 (Validation) under the proposed A5

For Justus, on `Writing/Methodology.docx` (4.4–4.8 are identical in
`Methodology_formatted.docx`). Comments only: each item quotes a sentence and
says why it becomes wrong or unsupported if A5 (`docs/decisions.md`) is
accepted. The text is not rewritten.

Figures: run `20261018T122536Z` (d100k, DeepSeek alone), labels
`20261017T182223Z`, from `results/metrics/validation_grid_20261018T122536Z.csv`,
`validation_selection_…csv` and `validation_design_…json`. A5's design is
member × Congress means, ≥ 10 substantive speeches, Nokken-Poole, read within
party × chamber × Congress cells.

## Wrong under A5

1. *"Therefore, for the validation, we exclude all speeches which every model
   of the run scored with exactly 0.0 on the ideology scale."*
   A5 excludes by **label**, as the trend analysis does ([A3]: procedural,
   ceremonial, unlabelled). The two rules are far apart. Of 99,902 validation
   speeches, the 0.0 rule drops 38,824, including **12,890 labelled policy
   debate**, and keeps 4,674 that the labels drop. "Every model of the run"
   also no longer describes the main run, which has one model, so the rule is
   "DeepSeek scored 0.0". The rule barely moves the headline (within-cell r
   0.54 with labels, 0.55 with the 0.0 rule). The labels are chosen for
   consistency with the trend analysis, not for r.

2. *"The excluded speeches stay in the data set for all other purposes."*
   Not since [A3]: the overview's trend analysis leaves out the same
   procedural and ceremonial speeches. Under A5 the validation and the trend
   analysis use one speech set; that is criterion 1. What does stay in every
   trend is the **unmatched** speeches (the sentence before this one is right
   about those).

3. *"The main statistic is the correlation between the scores from our models
   and the Nokken-Poole score within each party."* and *"Therefore, this is the
   one we report as the headline validation result."*
   A5 makes the headline the correlation **within party × chamber × Congress**:
   both parties pooled after each member-Congress is taken relative to its own
   party, chamber and Congress, with the per-party and per-chamber values
   beside it. A within-party r pooled over 13 Congresses also picks up the
   party's mean moving from Congress to Congress, in speeches and votes
   together. P8 shows exactly such moves (each party is most extreme in
   opposition). At A5's design: within D 0.46 → within-cell D 0.50; within R
   0.59 → within-cell R 0.57; pooled within-cell **0.54**.

## Unsupported, or needs a number

4. *"Members need a minimum number of scored speeches in the Congress to be
   included, because a mean of only one or two speeches would mostly reflect
   the topic of those speeches."*
   - The number is **10**, [A3]'s threshold for following members.
   - "Mostly reflect the topic" is not what was measured. What was measured is
     that a single speech is a noisy reading of its speaker's position:
     within a cell, only about a fifth of a single speech's variance is
     stable between members (ICC ≈ 0.18). How much of the rest is topic is
     not known.
   - "Scored speeches" should be the speeches kept after the exclusion (see 1).
     The same applies to *"the mean of all scored speeches in that Congress"*
     in the sentence before.

5. *"Across both parties, a high correlation would show that our created scores
   can tell Democrats from Republicans."*
   At A5's design the overall r is 0.92, while the party label alone
   correlates 0.94 with Nokken-Poole. A high overall r is what party alone
   produces, so it is weak evidence about the scores. "For completeness"
   (next sentence) is the right status. As evidence that the scores separate
   the parties, it is unsupported.

6. *"We use the Nokken-Poole scores, because our scores are also measured per
   Congress and members can change their positions over time."*
   The choice stands (A5, A). Two caveats it invites:
   - Nothing rides on it at this unit: career-long `nominate_dim1` gives the
     same within-cell r (0.54 with either).
   - The validation is **cross-sectional**. It shows members are ordered like
     their votes within a Congress, not that the scores track a member's
     **movement**. The change in a member's score from one Congress to the
     next follows the change in their Nokken-Poole score only weakly
     (within-cell r = 0.14 [0.08, 0.19] over 1,308 consecutive pairs). If the sentence is
     read as "the validation covers change over time", it is unsupported, and
     RQ2's member-level claims inherit that.

7. *"We therefore read the correlation as evidence that the scores follow the
   same underlying dimension, and not as a test of whether they replicate
   DW-NOMINATE."*
   This holds. It lacks the caveat from 6: the correlation is not evidence that
   the scores measure movement over time.

8. *"Since this scale runs in the same direction as our ideology score, … the
   two can be compared directly."*
   This holds for correlations, not for levels. One unit of our score
   corresponds to about 0.4 Nokken-Poole units (A5 regression). "Directly" is
   unsupported if it means "on the same scale".

## Missing: a choice the threshold forces

9. *(No sentence yet.)* The threshold selects. The sample gives every chamber
   the same quota of speeches, so a senator gets about four times a
   representative's speeches. At ≥ 10, **45% of the member-Congresses are
   senators, against 19% in Congress**, and they have served longer (mean
   tenure 9.1 / 7.0 Congresses for D / R, against 6.8 / 5.5). Per Congress,
   the kept units are the most frequent 15–30% of each party's
   representatives but 56–98% of its senators. The pooled r therefore leans
   on the Senate, and House and Senate need reporting apart.
   [A1] keeps chambers apart everywhere else too.

## Still correct

- *"…the confidence intervals are calculated by resampling members and not
  single speeches."* That is what the grid does (1,000 draws, stratified by
  party; a member's Congresses are drawn together). Do not quote the speech-level
  intervals `validate_scores.py` prints: they treat speeches as independent.
- *"The comparison on the level of single speeches is reported as a second
  view."* Supported.
- Pearson as the main statistic, with Spearman as a check: they agree within
  about 0.02 at A5's design.

## Outside O9, noticed while checking

- *"This link is made over the last name, party, chamber, Congress and state of
  the speaker."* On the Stanford side the crosswalk matches surname, state,
  Congress and chamber, and party only breaks ties ([D21], [D23]). On the
  govinfo side it joins on the bioguide id, Congress and chamber, with no
  names.

## 4.4–4.6 still describe a multi-model design (flagged, not fixed)

The main run, `d100k` (99,993 speeches), is **DeepSeek-V4.1-Flash alone**
([S19]). GPT-4o scored only the 5,200-speech S8 sample, and Claude Sonnet 4.6
only the two pilots ([S15]). Wording that still assumes several models:

- 4.3: *"…gives us the opportunity to compare the scores and to create an
  average that reduces the influence of any single model's bias…"*. The main
  results are not an average. [S19] asks the methodology to state what is
  lost (the cross-provider check).
- 4.4: *"For this thesis we initialy decided on the following models"*, the
  three model paragraphs and Table 4.1. All three are presented as the scoring
  set.
- 4.4: *"After showing which models we used in our analysis…"*. The analysis
  uses one.
- 4.4, DeepSeek: *"the model weights are open under the MIT license"*. CLAUDE.md
  marks this **[assumed]**, verified by secondary sources only. Also relevant
  to the fixed-version criterion: the model behind `deepseek-flash` changed on
  2026-10-17 under the same name ([S18]); all of d100k comes from the later
  build.
- 4.5: *"The same prompt … is used for all models and runs."*
- 4.5: *"These speeches are kept in the data, so they pull the mean scores
  towards the center. This will be addressed in the analysis."* Since [A3]
  they are left out by label in the overview analysis.
- 4.6: *"the final scores of our models"*, *"the scores from our models"*,
  *"every model of the run"* (see 1).

The evidence that a second model would tell the same validation story is now
on file: on the S8 speeches, at career level (≥ 5 substantive speeches, 119 +
119 members), DeepSeek (old build), GPT-4o, their mean and the new DeepSeek
build all give a within-cell r of about 0.49 (A5, robustness).
