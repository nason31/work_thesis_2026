# Appendix inventory — deep dive (notebooks 08–10)

2026-11-19 · Konstantin · decided in `docs/decisions.md` A11

Run `d100k` (`20261018T122536Z`), labels `20261017T182223Z`, substantive speech
(A3). Figures in `results/plots/` (thesis copies in `thesis/figures/`), tables
in `results/metrics/`; `<RUN>` = `20261018T122536Z`. Intervals: 95%, clustered
by member; bootstraps 1,000 draws, seed 42. House and Senate apart (O8). The
source break (2016-09-10, inside the 114th Congress) is not drawn on any figure
(A10, A11). No caption names it.

This file lists material and numbers only. It is not thesis text.

---

## 1. Congressional Review Act votes

- **Figure:** `deepdive_rollcall_undo.png` (bottom row: CRA roll calls, all and
  passed in that chamber; top row: share of roll calls to undo).
- **Tables:** `deepdive_rollcall_cra_per_congress_<RUN>.csv` (per Congress: title
  text against the wider rule), `deepdive_rollcall_cra_results_<RUN>.csv`
  (vote_result by chamber), `deepdive_rollcall_undo_<RUN>.csv` (per Congress ×
  chamber: `n_cra`, `n_cra_passed`, `n_resolutions_voted`,
  `n_resolutions_passed_a_chamber`), hand check
  `deepdive_keyword_check_rollcall_undo.csv`.
- **Numbers:**
  - Wider rule (A9) 237 CRA roll calls (House 78, Senate 159), against 184
    with the title text alone; the title text misses the older phrasings
    (107th–109th: 0 against 6).
  - Per Congress (wider rule): 0–7 up to the 113th, then 18 (114th), 44
    (115th), 9, 15, 48 (118th), 87 (119th, 2025 only).
  - Distinct resolutions passing at least one chamber: 6 (114th), 19 (115th),
    2, 8, 20 (118th), 26 (119th).
  - vote_result: House 70 Passed, 8 Failed; Senate 74 Joint Resolution Passed,
    3 Bill Passed, 12 Defeated, 38 Motion to Proceed Agreed to (not passage),
    5 Veto Sustained, the rest procedural.
- **Passing one chamber is not overturning a rule.** "Passed" counts the vote
  in that chamber; the other chamber and the president can still stop the
  resolution. In the 118th, 11 votes to override a presidential veto of such
  resolutions failed (6 House, 5 Senate, A9 reading of section 10).
- Undo keyword precision: 45 of 50 hand-checked hits (90%) before the
  exclusions; the exclusions remove all 5 false hits and none of the 45.

## 2. Positions against topic mix (Shapley decomposition)

- **Figures:** `deepdive_gap_decomposition_107_118.png` (107→118 only; main
  text), `deepdive_gap_decomposition.png` (107→118 and 107–114 vs 115–119),
  `deepdive_steps_decomposition.png` (every step; main text),
  `deepdive_switch_decomposition.png` (the four changes of the White House).
- **Tables:** `deepdive_gap_decomposition_<RUN>.csv`,
  `deepdive_steps_decomposition_<RUN>.csv`, `deepdive_steps_by_period_<RUN>.csv`,
  `deepdive_switch_decomposition_<RUN>.csv`.
- **Numbers, 107th → 118th (2001–2023), policy debates:**
  - House: gap 0.541 → 0.962, change +0.421 [+0.342, +0.496] = positions
    +0.370 [+0.306, +0.435] + agenda +0.050 [+0.024, +0.079].
  - Senate: 0.475 → 0.945, change +0.470 [+0.363, +0.577] = positions +0.483
    [+0.367, +0.594] + agenda −0.013 [−0.053, +0.034].
  - 107–114 against 115–119: House +0.194 (positions +0.170, agenda +0.024);
    Senate +0.320 (positions +0.323, agenda −0.003).
- **Steps (107→119):** the four switch steps add House +0.60, Senate +0.72,
  more than the whole growth (+0.53, +0.63): 110→111 +0.196 / +0.212, 114→115
  +0.051 / +0.210, 116→117 +0.244 / +0.135, 118→119 +0.109 / +0.161. Within
  presidencies: Bush +0.137 / +0.145, Obama −0.115 / −0.057, Trump I −0.071 /
  −0.097, Biden −0.021 / −0.080 (House / Senate).
- Split exact by construction (two-factor Shapley, no residual); a period's G
  comes from averaged shares and means (A7).

## 3. Further event windows (Iraq War, Affordable Care Act, Dobbs)

- **Figures:** `deepdive_events_change_gap.png`, `deepdive_events_change_tone.png`,
  `deepdive_event_blocks_{gap,agenda,tone}.png`.
- **Tables:** `deepdive_event_windows_<RUN>.csv`, `deepdive_event_cell_sizes_<RUN>.csv`,
  `deepdive_event_effects_<RUN>.csv`, `deepdive_event_blocks_<RUN>.csv`,
  `deepdive_event_blocks_cell_sizes_<RUN>.csv`.
- **Numbers** (after − before; † = a cell under 30 speeches):
  - **Iraq War** (2003-03-20; window 76 days a side, cut at the 108th's start):
    no clear change in the gap in either chamber; House Republicans' tone in
    defence/foreign −0.185 [−0.357, −0.014] †; Senate Republicans' tone in
    other topics −0.147 [−0.215, −0.078].
  - **ACA signed** (2010-03-23; 182 days): House gap in health −0.307
    [−0.500, −0.114] †, −0.349 [−0.564, −0.134] † beyond other topics; House
    Democrats' tone in health −0.135 [−0.251, −0.019] † (−0.208 beyond other
    topics); Senate Democrats' tone in health −0.113 [−0.217, −0.009] †.
  - **Dobbs** (2022-06-24; 182 days): no clear change in the gap; House
    Democrats' tone in social-cultural +0.097 [+0.010, +0.183], +0.108
    [+0.009, +0.207] beyond other topics.
- 13 of 96 cells under 30 speeches (A8); Iraq and Lehman are the weakest
  cases.

## 4. Windows after elections

- **Figure:** `deepdive_post_election.png`.
- **Tables:** `deepdive_post_election_windows_<RUN>.csv`,
  `deepdive_post_election_losers_<RUN>.csv`, `deepdive_post_election_effects_<RUN>.csv`,
  `deepdive_post_election_cell_sizes_<RUN>.csv`.
- **Numbers:**
  - 12 elections × 2 chambers = 24; 5 midterm chambers without a change of
    majority (House 2002, 2014; Senate 2010, 2018, 2022) → 19 with a loser;
    **18** with an estimate (House 2008 blank: fewer than two members of a
    side speak in both windows).
  - **47 single cases:** each party's own change per election × chamber, 47 of
    48 estimable (House Republicans 2008 blank), per outcome.
  - Pooled, losers beyond winners: tone presidential House +0.065 [−0.016,
    +0.147], Senate +0.026 [−0.030, +0.081]; midterm +0.035, +0.045 (not
    clear). Extremity: only Senate midterms +0.249 [+0.112, +0.385].
  - Clear single elections: tone Senate 2016 +0.233 [+0.022, +0.445];
    extremity Senate 2006 +0.253 †, 2014 +0.253, 2016 +0.194.
  - Trend per decade: only House presidential extremity +0.161 [+0.016,
    +0.305].
  - Windows 2–27 sitting days a side; 9 of 96 party × period cells under 30.

## 5. Event study for extremity, with its failed pre-trend check

- **Figures:** `deepdive_wh_event_study_extremity.png` (appendix);
  `deepdive_wh_event_study_tone.png` (main text); `deepdive_wh_event_study.png`
  (both).
- **Table:** `deepdive_wh_event_study_<RUN>.csv` (`kind` = effect / own path).
- **Numbers, extremity, losing − winning party, relative to k = −1:**
  - Pre-trend k = −2: House −0.041 [−0.078, −0.005], Senate +0.068 [+0.029,
    +0.106] — **the check fails in both chambers, with opposite signs** (A9).
  - k = 0: House +0.106 [+0.083, +0.130], Senate +0.124 [+0.083, +0.165];
    k = +1: House +0.066, Senate +0.222.
  - For comparison, tone: pre-trend House −0.023 [−0.054, +0.009], Senate
    −0.007 [−0.048, +0.035]; k = 0 House +0.142, Senate +0.184.
- For extremity the thesis relies on the chamber flips (section 3 of notebook
  10, placebos at zero).

## 6. Keyword lists, hand checks and precision

- **Tables:** `deepdive_keyword_lists.csv` (52 terms in 4 lists: cooperation,
  hostility = v2, deficit, undo), `deepdive_keyword_precision_<RUN>.csv`, the
  hand-check sheets `deepdive_keyword_check_{cooperation,deficit,other_party,undo,hostility,hostility_v2,presidential_power,rollcall_undo}.csv`
  (definition of a true hit in each sheet's first line), counts
  `deepdive_keyword_counts_<RUN>.csv`, `deepdive_keyword_counts_corpus_<RUN>.csv`.
- **Not in `deepdive_keyword_lists.csv`:** the `other_party` terms
  (`deep_dive.OTHER_PARTY_TERMS`, by the speaker's party) and the exclusion
  phrases (`deep_dive.KEYWORD_EXCLUDE`); both are written out in A8.
- **Precision (50 random hits each, seed 42, one hit per speech):**
  cooperation 43/50 = 86%, deficit 49/50 = 98%, other_party 49/50 = 98%,
  undo 48/50 = 96%, hostility v2 40/50 = 80% (used); hostility v1 28/50 = 56%
  (replaced), presidential_power 18/50 = 36% (dropped); roll-call undo 45/50
  = 90% before the exclusions.

## 7. Models A, B and C side by side

- **Figures:** `deepdive_opposition_models_extremity.png`,
  `deepdive_opposition_models_tone.png` (A, B, C); `deepdive_opposition_models_model_c.png`
  (C only; main text).
- **Tables:** `deepdive_opposition_models_<RUN>.csv`,
  `deepdive_opposition_identifying_<RUN>.csv`, `deepdive_opposition_robustness_<RUN>.csv`
  (C without party switchers, without Specter's miscoded speeches).
- **Out of the White House, A / B / C:**
  - Extremity: House D +0.004 / +0.004 / +0.019, House R +0.110 / +0.109 /
    +0.089, Senate D +0.093 / +0.070 / +0.069, Senate R +0.130 / +0.107 /
    +0.096.
  - Tone: House D +0.065 / +0.063 / +0.062, House R +0.076 / +0.074 / +0.070,
    Senate D +0.141 / +0.100 / +0.103, Senate R +0.099 / +0.067 / +0.070.
- **Chamber minority, B / C:** extremity +0.028 / +0.017 (House D), +0.031 /
  +0.003 (House R), +0.035 / +0.031 (Senate D), +0.035 / +0.015 (Senate R);
  tone +0.136 / +0.124, +0.084 / +0.073, +0.065 / +0.054, +0.049 / +0.031.
- **Identifying members (C):** opposition House D 378 of 580, House R 402 of
  695, Senate D 86 of 123, Senate R 85 of 132; chamber minority 357, 385, 102,
  114.
- A reproduces notebook 07's `findings_opposition_effect` and
  `findings_tone_opposition_effect` by assert.

## 8. The opposition's tone per presidency (escalation)

- **Table:** `deepdive_opposition_escalation_<RUN>.csv` (from
  `findings_tone_cells_<RUN>.csv`; unweighted mean of the per-Congress party
  means). Figure behind it: `findings_tone.png`.
- **Numbers (out of the White House / governing):**

  | Presidency | Congresses | Senate | House |
  |---|---|---|---|
  | Bush | 107–110 | 0.29 / 0.25 | 0.38 / 0.29 |
  | Obama | 111–114 | 0.41 / 0.31 | 0.37 / 0.35 |
  | Trump I | 115–116 | 0.52 / 0.36 | 0.37 / 0.31 |
  | Biden | 117–118 | 0.58 / 0.36 | 0.50 / 0.38 |
  | Trump II | 119 (2025) | 0.65 / 0.51 | 0.59 / 0.42 |

- Reproduces the values quoted in the analysis, by assert (notebook 10,
  section 11).

## 9. Robustness: the days before the inauguration

- **Table:** `deepdive_robustness_january_<RUN>.csv` (variant, analysis, term,
  party, chamber, outcome, estimate, interval, p, the main estimate with its
  interval and p, speeches dropped in total and in the chamber).
- **Design:** speeches dated 3–19 January dropped; A = 2001, 2021 (101
  substantive speeches: House 81, Senate 20); B = 2001, 2009, 2017, 2021, 2025
  (634: House 393, Senate 241). Model C, power ladder, White House event study,
  tone and extremity, existing functions and defaults; the main estimates
  reproduce the committed tables by assert (60 estimates).
- **Numbers:**
  - No sign change; largest shift 0.008 (A), 0.009 (B); mean 0.001, 0.003.
  - One significance change, in B: Senate R tone, chamber only (out of the
    White House), +0.032 [−0.001, +0.066] → +0.034 [+0.000, +0.068].
  - Event study tone, k = 0: House +0.142 → +0.145 (A) / +0.148 (B); Senate
    +0.184 → +0.190 / +0.193. Tone pre-trend stays at zero.
  - Model C tone, out of the White House: +0.062 to +0.103 → +0.064 to +0.104.
  - Power ladder tone, nothing: +0.101 to +0.180 → +0.101 to +0.188.

## 10. Tone of majority against minority, House, economic debates

- **Figure:** `deepdive_tone_majority_minority.png` (notebook 10, section 13; main
  text).
- **Source:** `deepdive_tone_by_topic_<RUN>.csv` (09's tone model by block),
  `chamber == "H"`, `topic_block == "economic"`; each party labelled minority or
  majority from `deep_dive.HOUSE_MAJORITY` (`majority_minority_tone`).
- **Assert:** the minority is harsher (higher mean) in 12 of 13 Congresses; the
  one exception is the 118th (majority R 0.462, minority D 0.439).

## 11. Replacement against adaptation, 2001–2025

- **Figure:** `deepdive_turnover_2001_2025.png` (the window 107th–119th only;
  main text); both windows in `deepdive_turnover.png`.
- **Table:** `deepdive_turnover_cumulative_<RUN>.csv`, window
  "2001–2025 (107th–119th)"; robustness `deepdive_turnover_robustness_<RUN>.csv`.
- **Numbers (total = adaptation + replacement, small unmatched part aside):**
  House D −0.243 [−0.305, −0.184] = −0.172 + −0.071; House R +0.269 [+0.208,
  +0.325] = +0.054 + +0.212; Senate D −0.304 [−0.354, −0.259] = −0.241 +
  −0.063; Senate R +0.308 [+0.218, +0.407] = +0.137 + +0.171.
- Replacement rests on within-Congress differences (validated, r = 0.54),
  adaptation on change over time (r = 0.14, A5, A7).
