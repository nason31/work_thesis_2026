# Deep dive — draft decisions A7 and A8

Drafts for `docs/decisions.md`, not yet entered. A7 covers notebook 08, A8
notebook 09. To be entered by the team; the "Findings" of A8 are filled in once
it is settled what goes into the thesis.

**Offen:** Linie zum Quellenwechsel (2016-09-10) aus den Deep-Dive-Plots entfernen.
Wartet auf Änderung von CLAUDE.md/plots.py durch das Team, danach 08 und 09 neu laufen lassen.
Hinweis: Vier Funktionen in `deep_dive_plots.py` (`plot_yearly_keyword_lines`, `plot_common_ground`,
`plot_laws_and_gap`, `plot_keyword_lines_by_congress`) rufen `mark_source_break` selbst auf.
Wenn das Team die Linie entfernt, müssen diese Aufrufe auch raus, sonst bleibt die Linie
oder die Funktionen brechen ab. Außerdem in Notebook 09, Zelle 1, den Satz
"the 2016-09-10 source break is marked in every time plot" anpassen.
Notebook 10 muss nach der Änderung ebenfalls neu laufen; dort rufen auch `plot_steps_decomposition`,
`plot_rollcall_lines` und `plot_rollcall_undo` `mark_source_break` selbst auf, und die Kopfzelle
erwähnt die Linie.

**Offen (später, in einem gemeinsamen Durchgang mit dem Entfernen der Linie):**
- Untertitel der Deep-Dive-Plots umbrechen: Die langen einzeiligen Untertitel machen die PNGs
  sehr breit (z. B. in Notebook 10 rund 4.500 px), die Panels rücken dadurch nach links.
- Beim Neulauf das Überschreiben geänderter Plots gezielt erlauben: Die Vergleichszelle in
  Notebook 10 stoppt sonst bei jedem geänderten PNG.
- Tabellen weiter mit Toleranz prüfen (Regel unter A9, „Reruns“).

---

### A7 — The deep dive (notebook 08): chamber control, opposition within members, topic model, two decompositions
**Date:** 2026-10-27 · **Status:** proposed · **Decided by:** Konstantin; implementation by the agent

`code/notebooks/08_deep_dive.ipynb` on `d100k` ([S19]) and its labels ([L8]), substantive speech ([A3]).
Logic `code/src/deep_dive.py`, figures `code/src/deep_dive_plots.py`, tests `code/tests/test_deep_dive.py`.
Outputs `results/plots/deepdive_*.png`, `results/metrics/deepdive_*_20261018T122536Z.csv`.

- **Chamber majority per Congress, by date where the Senate changed hands.** House:
  history.house.gov, Party Divisions. Senate: senate.gov Party Division (checked via the table citing it;
  senate.gov refuses automated requests). 107th Senate: D (Gore) to 2001-01-20, R (Cheney), D from
  2001-06-06 (Jeffords; announced 2001-05-24). 117th: R to 2021-01-20, then D. `chamber_minority` = the
  speaker's party is not the majority of their chamber that day. **Rejected:** majority per Congress only
  (misassigns the 107th and early 117th Senate); a unified-government indicator (not needed for the
  question, dropped).
- **Constellations** (White House yes/no × chamber majority yes/no): means of **extremity** (ideology
  toward the party's own side, D sign-flipped) and tone per party × chamber, pooled over Congresses.
  **Rejected:** raw ideology (D and R would move in opposite directions; not comparable with 07).
- **Opposition models, per chamber × party, extremity and tone, member-clustered:** A = notebook 07's
  (party-chamber intercept, trend, out of the White House; must reproduce the committed tables), B = A +
  chamber minority, C = B with member fixed effects. C is identified only by members whose status changes;
  their number is reported. Robustness of C without party switchers ([D23]) and without Specter's
  miscoded speeches ([O13]).
- **Topic model:** per chamber, saturated OLS of ideology on party × block × Congress, Congress as a
  category, member-clustered; gaps equal cell means; per-block test 107th vs 118th (last complete
  Congress) and a joint Wald test of equal change across blocks; stops if a block lacks a cell. Cells
  under 30 speeches are reviewed first. **Rejected:** a linear trend per block (assumes steady change);
  coarse periods (hide the steps of [A6]).
- **Agenda vs. positions:** the change in the policy-debate gap, G = Σ s^R μ^R − Σ s^D μ^D with each
  party's own shares, split by two-factor Shapley (no residual); per-block agenda against the party's own
  mean; 107→118 and 107–114 vs 115–119. For a period, shares and means are averaged over its Congresses
  (equal weight) and G is computed from those averages: the split is exact for that G, which can differ
  in the third decimal from the mean of per-Congress gaps. Member bootstrap, 1,000 draws, seed 42.
  Sample shares within cells are valid since each cell is a random draw with equal quota ([S11]).
  **Rejected:** one reference order (leaves an interaction term or depends on the order).
- **Replacement vs. adaptation:** per chamber × party and consecutive Congresses,
  ΔM = (m_s′ − m_s) + w_n(m_n − m_s′) − w_x(m_x − m_s) + unmatched term, an identity; speech-weighted
  (the run draws speeches); stayers/entrants/exits by actual Voteview membership (delegates excluded,
  [D10]); chamber moves and party switches with a new ICPSR count as exit + entry; speeches without
  ICPSR kept as their own term (CLAUDE.md: never drop from the trend). A member relabelled without a new
  ICPSR (Specter, other Stanford switchers) counts as a stayer, so a robustness table repeats the split
  without party switchers and without Specter's miscoded speeches; the main split keeps every speech to
  match 07. **Rejected:** member means (too few members with enough speeches); membership by presence in
  the sample; defining stayers by Voteview party (independents would exit and re-enter every Congress).
- **Validation caveat:** replacement rests on within-Congress differences (within-cell r = 0.54, [A5]);
  adaptation on change over time (r = 0.14, [A5]). State it wherever the split is reported.

---

### A8 — The deep dive extensions (notebook 09): events, cohorts, dissent, keyword lists, topics, tone, laws
**Date:** 2026-11-03 · **Status:** active · **Decided by:** Konstantin (analyses, event list, keyword lists, hand check, all choices marked "decided"); implementation by the agent

`code/notebooks/09_deep_dive_extensions.ipynb`, on the `d100k` run ([S19]) and its labels ([L8]), substantive speech ([A3]); topics on policy debates only. Logic in `code/src/deep_dive.py`, figures in `deep_dive_plots.py`, tests in `test_deep_dive.py`. Figures `results/plots/deepdive_*.png`, tables `results/metrics/deepdive_*_20261018T122536Z.csv`. Chambers never pooled ([O8]); intervals clustered by member; member bootstraps 1,000 draws, seed 42.

**Time window.** Score-based analyses run to the end of the scores (2025-12-18); text analyses to the end of the corpus (2025-12-19, [D15]). Nothing is fixed to the 119th Congress: the last Congress, the last complete Congress (data reaching 1 December of its second year; now the 118th) and an incomplete last year are read from the data, so a later run needs only a new run id. Not rebuilding the corpus to reach 2026 (would change [D15]).

**A. Events.** Fixed list, set before any result: 9/11 (2001-09-11, defense_foreign), Iraq War (2003-03-20, defense_foreign), Lehman (2008-09-15, economy_tax_budget), ACA signed (2010-03-23, health), Covid emergency (2020-03-13, health + economy_tax_budget), Dobbs (2022-06-24, social_cultural). Window: six months before against six months after; never across a Congress boundary — both sides cut to the nearer boundary (Iraq 76 days, accepted; Lehman 110). One saturated model per event and chamber (party × period × event topic/other), member-clustered: change in R − D and in each party's tone, in the event's topics, in the rest, and the difference. 13 of 96 cells under 30 speeches (11–28 speeches, 9–21 members) are kept and marked †; only clear intervals are read; Iraq and Lehman are the weakest cases. Quarterly view only where every party × bin cell holds ≥ 30 (none does). *Rejected:* windows across Congress boundaries; dropping events with small cells.

**B. Cohorts.** First Congress in either chamber (`member_background`, [A4]), from the 107th; extremity in the first Congress relative to party × chamber × Congress. ICPSRs with earlier service under another ICPSR (party switchers renumbered by Voteview; matched by bioguide id) left out (17 with speeches).

**9. Dissenters.** Units: member × Congress, ≥ 10 substantive speeches ([A3], [A5]). Moderate = member mean closer to the other party's median member than to the own party's (same chamber and Congress); robustness |mean| < 0.2. *Rejected:* the innermost fifth (20% by construction). Exit = not in the same chamber next Congress (Voteview service); last Congress left out; linear probability model within Congress.

**4. Last Congress.** Notebook 08's model C plus `last_congress` = the person (bioguide id) sits in neither chamber next Congress; a House → Senate move is not leaving; the data's last Congress left out. Whether a member retired or lost is not known.

**Keyword lists.** Whole words/phrases, case ignored, inflections written out, exclusion phrases blanked first. Checked by hand, 50 random hits per list from substantive `d100k` speeches (seed 42, one hit per speech; definition of a true hit in each sheet). Precision: cooperation 86%, deficit 98%, other_party 98%, undo 96%; presidential_power 36% — **dropped** ("unconstitutional" mostly about laws and courts, "power grab"/"abuse of power" often about Congress); hostility v1 56% — **replaced** ("radical"/"extremist" often non-political or about terrorism); hostility v2 80% — used (most false hits: "socialism" in denials, general statements, quotations). Lists:
- cooperation: bipartisan, bipartisanship, compromise(s), common ground, across the aisle, both sides of the aisle, work/working/worked together, both parties; excluded "compromise … security".
- hostility (v2): socialist(s), socialism, radical left/right, far left, far-left, far right, far-right, radical agenda, radical policy/policies, extremist agenda, extreme agenda, environmental extremists/radicals, radical environmentalists/environmentalism; excluded Soviet socialist, national socialist(s), socialist republic(s).
- other_party: D speakers Republicans, Republican Party, GOP, my Republican colleagues, Republican leadership/majority/leader(s); R speakers Democrats, Democratic Party, Democrat Party, my Democratic colleagues, Democratic/Democrat leadership, Democratic majority, Democratic leader(s), Democrat leaders; both: the other side of the aisle, my friends/colleagues on the other side, those/members on the other side. Bare "Republican"/"Democratic" and the own party never count.
- deficit: deficit(s), national debt, debt ceiling, debt limit, fiscal responsibility, balanced budget; excluded trade deficit(s), attention deficit, deficit(s) of trust/confidence.
- undo: repeal(s/ed/ing), overturn(ed/ing), undo/undoing/undid, roll(s/ed/ing) back, rollback(s).

Main basis: substantive `d100k` speeches; robustness: the whole corpus (all speech types, read in batches). Hits per 10,000 words beside every share ([D17] changes speech length at the break). Cooperation and hostility shown per Congress (hostility too rare for yearly lines); yearly values in the table. Power words: opposition effect net of the trend ([A6]'s model A on the 0/1 hit). *Note:* [L1] rejected keyword dictionaries for topic labels; here they are rhetoric markers with a reported precision.

**13. Swing states.** MIT Election Lab county returns 2000–2024 (Harvard Dataverse, doi:10.7910/DVN/VOQCHQ, V20, CC0): a county's TOTAL row where present, its modes summed otherwise (checked: PA 2020 = 0.506, FL 2000 = 0.500). Margin = mean |D two-party share − 0.5| over the last two presidential elections before the Congress's first year (107th–108th: 2000 only, `n_elections` = 1). Terciles of states per Congress; Senate only — no district data. Linear term per 10 points within party × Congress.

**D, E.** Notebook 08's decomposition for every change of the White House (read from `PRESIDENTS_BY_CONGRESS`: 110→111, 114→115, 116→117, 118→119), 107→118 run with it and reproducing 08 (values and intervals). Topic tests 107 → last Congress (main; reverses 08's choice of the 118th as the test Congress) and 107 → last complete Congress (control, reproduces 08).

**1. Common ground.** Per Congress × chamber (× block): share of D speeches > R median, of R speeches < D median (strict), and the overlap coefficient over observed scores (exact on the 0.05 grid); overall on substantive speech, blocks on policy debates; member bootstrap.

**G.** Notebook 08's topic model on tone; change per party and block, joint Wald test per party. Tone and ideology come from the same model.

**3.** Tone with vs. without naming the other party, within party × chamber × Congress.

**11. Laws.** Public laws per Congress (congress.gov, retrieved 2026-11-02; include ceremonial laws). Pearson r with Fisher-z interval over complete Congresses (12), and after a linear trend is removed from both series; the 119th left out. Descriptive only.

**Findings (preliminary)** are in the notebook's reading cells.

---

### A9 (Entwurf) — Deep dive designs, topics over time and votes (notebook 10)
**Date:** 2026-11-03 · **Status:** proposed · **Decided by:** Konstantin (analyses, cases, independents, terciles, HALT 2 rules and hand check); implementation by the agent

`code/notebooks/10_deep_dive_designs_and_votes.ipynb`, on the `d100k` run ([S19]) and its labels ([L8]), substantive speech ([A3]); topics on policy debates only. Logic in `code/src/deep_dive.py`, figures in `deep_dive_plots.py`, tests in `test_deep_dive.py`. Figures `results/plots/deepdive_*.png`, tables `results/metrics/deepdive_*_20261018T122536Z.csv`. Intervals clustered by member; member bootstraps 1,000 draws, seed 42. Time window as A8: scores to their end, the last and last complete Congress read from the data; Voteview roll calls cut at 2025-12-31 ([D15]), the 119th marked incomplete. Presidents and chamber majorities only from `PRESIDENTS_BY_CONGRESS`, `HOUSE_MAJORITY`, `SENATE_MAJORITY`, `SENATE_MAJORITY_CHANGES`.

**1. Policy blocks.** A8's topic model; change of R − D (and of each party's tone) from the 107th to the last Congress (main) and the last complete one (control). Joint Wald test that the four policy blocks changed alike (economic, social_cultural, security_foreign, governance; "other" left out, df = 3) and all six pairwise differences with Holm-adjusted p. Reproduces A8's block changes exactly.

**2. Steps.** A7's decomposition (positions against agenda, two-factor Shapley) for every step from one Congress to the next, per chamber; one member bootstrap for all steps, whose draws do not depend on the steps, so the switch steps reproduce A8's intervals. Per block two quantities are reported: the *contribution* (positions + agenda, weighted by the block's share of debates) and the *within* change of R − D (unweighted). Periods: steps within a presidency (`add_presidency`'s term) and each change of the White House alone; summed values with percentile intervals, shares of the period's change and of the whole growth, joint Wald test of the four policy blocks with bootstrap covariance (df = 3). *Rejected:* comparing only endpoints (hides when growth happened).

**3. Chamber flips (difference-in-differences).** Cases read from the majority constants: t → t+1 with exactly one chamber changing hands and the president's party unchanged: 111→112 House, 113→114 Senate, 115→116 House, 117→118 House (main); 107→108 Senate robustness only (the 107th Senate changed hands within the Congress). Treated: members of the party losing the flipping chamber, in it; control: the same party in the other chamber. Member fixed effects + post + post × treated per case; pooled with member × case and case × post fixed effects; extremity and tone. Main: members in the same chamber in both Congresses (Voteview service); robustness: all members, the 114th with govinfo speeches, the 107→108 case. In the 114th only Stanford speeches (to 2016-09-09) in the main variant, so no main case crosses the source break (decided 2026-11-03). Mirror: the party gaining the chamber. Placebo t−1 → t; those of 115→116 (crosses the break) and 117→118 (the control gained the Senate in the 117th) are not clean and not read. **O8:** compatible (decided 2026-11-03): member fixed effects remove levels, case × post the shared change; what remains compares two changes within one source; each chamber's own change is shown beside the estimate. *Rejected:* separate chambers without a joint estimate.

**4. White House event study.** Changes from `white_house_changes` (110→111, 114→115, 116→117, 118→119); k = Congress − the new president's first, −2 … +1 where available, reference k = −1. Treated: the party losing the White House; control: the party winning it. Stacked; member × change and change × k fixed effects, treated × k effects, `chamber_minority` as a control; per chamber, extremity and tone. k = −2 is the pre-trend test. At 114→115 the source break lies between k = −1 and 0; the party comparison removes only a source effect common to both parties. Each side's own path within members is shown descriptively.

- Thesis berichtet die Event Study (Analyse 4) nur für den Ton. Die Extremität kommt in den Anhang, weil der Vortrend-Test (k = −2) in beiden Kammern scheitert (House −0,04, Senat +0,07). Für die Extremität stützen wir uns auf Analyse 3 (Kammerwechsel, Placebos bei null).

**4b. Power ladder.** Four levels of the speaker's party's power, from the constants (`PRESIDENTS_BY_CONGRESS` per Congress; `HOUSE_MAJORITY`, `SENATE_MAJORITY`, the 107th and 117th Senate by date as `add_chamber_control`): full power (White House and the majority of the own chamber; reference), White House only, chamber only, nothing — the four `CONSTELLATIONS` of A7, now within members. Per party × chamber, A7's model C with one indicator per level (member fixed effects, linear trend in the Congress), clustered by member; tone and extremity. Extra test: β(nothing) − β(White House only) − β(chamber only) = 0, estimated as the interaction `opposition` × `chamber_minority` (the same model reparametrized; Wald χ²(1)). Cells: speeches, members and member × Congress units per level; a level under 30 speeches or 10 members is marked (none is). Background table per Congress and majority period: president, House and Senate majority, the president's constellation (unified, one chamber, none). The president is the constant's for the whole Congress, as in A6/A7: 3–20 January 2001 and 2021 carry the incoming one (101 substantive speeches). **Check:** the model with the two additive indicators reproduces A7's model C (`deepdive_opposition_models`), by assert. Findings: see the notebook.

**4c. After elections.** All general federal elections 2002–2024; the date by rule (the Tuesday after the first Monday in November), checked against the list 2002-11-05 … 2024-11-05 by assert. Windows in sitting days per chamber, i.e. days with at least one speech of that chamber in the corpus (decided at HALT 1, 2026-11-06; *rejected:* the days of the scored sample, which depend on the draw and the labels): after = election day to the end of the Congress (lame duck), before = as many sitting days right before election day; no window crosses a Congress boundary. 2016, House: the before window stops at the source break (10 sitting days before, 13 after; decided at HALT 1); robustness with the three Stanford days. Losers per election and chamber from the constants: presidential years, the party that does not win the presidency (at a re-election the challenger's), both chambers; midterms, the party losing the chamber's majority; midterm chambers without a change (House 2002, 2014; Senate 2010, 2018, 2022) are described (each party's own change), not pooled. Model: after − before, loser against winner, within members (member × election and election × period fixed effects), clustered by member; per election, pooled per election type (presidential, midterm) and chamber, and the change of the pooled effect per decade; tone and extremity; descriptive only. Cells under 30 speeches kept and marked †, only clear intervals read; an estimate is blank where fewer than two members of a side speak in both windows (`ELECTION_MIN_MEMBERS_BOTH`, confirmed 2026-11-06: with one member the change is fitted exactly and the clustered SE collapses towards 0; only House 2008 is affected); estimates resting on few members are read with their cell sizes. The weeks after an election have few sittings; the 2026 election and the 120th Congress lie outside the data. Findings: see the notebook.

**5. Naming the other party.** A8's `other_party` list (precision 98%), 0/1 per speech; linear probability model as A7's model C (member fixed effects, trend, `opposition`, `chamber_minority`) per party × chamber. Main: substantive `d100k`; robustness: the whole corpus.

**6. Swing states, signed.** `own_margin` = own party's two-party presidential share − 0.5, mean of the last two elections before the Congress (same data and rule as A8; 107th–108th: 2000 only). Linear per 10 points within party × Congress; terciles over the party's senators per Congress (decided 2026-11-03; *rejected:* terciles of the 50 states, which mirror between parties and leave near-empty cells); A8's absolute margin in the same table (reproduced). Within the same senator: senator fixed effects plus the party's Congress effects; number of senators with varying margin and its range reported. Voters' preferences and competition cannot be separated.

**7. Blocks around events.** A8's events and windows. Per event × chamber and block, before against after: the block's share of each party's policy debates (LPM), the change of R − D and of each party's tone within the block (saturated party × period × block model). Cells under 30 kept and marked †; contrasts with an empty cell left out.

**8. New against continuing members.** A8's definition of new (first Congress in either chamber, from the 107th, party switchers with a new ICPSR left out entirely). New − continuing within party × chamber × Congress: per cohort, pooled per party × chamber, per presidency. A8's comparison with the party mean (which contains the newcomers) in the same table. *Rejected:* the party mean as the only reference.

**9. Centrists.** Share of member × Congress units (≥ 10 speeches) with |mean| < 0.15, 0.2, 0.25; the bootstrap of A8's `dissent_table` draw for draw, so 0.2 reproduces its `share_centrist`. Moves mechanically with the gap; not an independent measure.

**10. Voteview.** Source: Voteview (Lewis, Poole, Rosenthal, Boche, Rudkin, Sonnet), `HSall_votes.csv`, `HSall_rollcalls.csv`, `HSall_members.csv`, voteview.com/data, downloaded 2026-11-03; 107th Congress on, roll calls to 2025-12-31. Cast codes 1–3 yea, 4–6 nay, 7–9 and 0 not counted. Party from the member file (100 D, 200 R). **Independents (decided 2026-11-03, as [D2]/[D18]):** with their caucus — Sanders, Jeffords, King, Manchin, Sinema → D, Goode (107th) and Kiley (119th) → R; Barkley, Amash, Mitchell excluded; delegates left out ([D10]); an independent no rule covers stops the build ([D4]); robustness without any independent. *Rejected:* dropping independents in the main variant (rejected in [D2]). Presidents' positions and ICPSRs that are not members that Congress left out and counted.
- *Party-unity vote:* a majority of D against a majority of R (Congressional Quarterly); *bipartisan:* both majorities on the same side; a party with equal yeas and nays has no majority (counted); *near-unanimous:* ≥ 90% or ≤ 10% yea. Shares among all roll calls (main); robustness without near-unanimous votes, only final passage, without independents.
- *Final passage (HALT 2):* a bill or joint resolution (HR, S, HJRES, SJRES) and a passage-type question — passage, suspension and passage, concurring in or agreeing to the other chamber's amendments, conference report, the Senate's vote on a joint resolution, veto override — and no procedural motion (recommit, table, instruct, reconsider, commit, refer, proceed, cloture, previous question, consideration). Corrections: "Disagree to the Senate Amendment" out, "Consideration of Conference Report" out, the typo "Supend the Rules and Pass" in. En-bloc suspensions without one bill number stay out. 5,379 final passage votes (House 4,695, Senate 684).
- *Correlation with the speech gap:* Pearson r with Fisher-z interval over the complete Congresses, raw and detrended (as the laws in A8). Descriptive.
- *Votes to undo:* whole words repeal(s/ed/ing), disapproval/disapproving, overturn(s/ed/ing), nullify/nullifies/nullified in vote_desc, vote_question or dtl_desc (not rescission). Procedural votes on undo measures count; robustness: share among final passage votes. Hand check: 45 of 50 random hits correct, **90% before the exclusions**; the five false hits (disapprovals of arms sales or exports, an expression of disapproval, a presidential report) led to exclusions blanked before matching — disapproval of a proposed foreign military sale / export / sale, of a presidential report, and "expressing disapproval" — anchored on the disapproval they qualify (blanking the phrases alone would leave the keyword). They remove all five false and none of the correct hits; 554 → 526 hits. No new sample.
- *Congressional Review Act (HALT 2):* a joint resolution (HJRES, SJRES) whose text, after the exclusions, names "chapter 8 of title 5" or a disapproval with "rule" within 40 characters, not a disapproval of a DC Council act. 237 roll calls against 184 with the title text alone (which misses older phrasings, e.g. the 2001 ergonomics resolution). Passed: "Passed" (House), "Joint Resolution Passed" or "Bill Passed" (Senate). Counted per roll call and as distinct resolutions passing at least one chamber; passing one chamber is not becoming law.
- Which roll calls happen is decided by the majority; bills without a majority often never come to a vote.

**Reruns: comparing with existing outputs (decided 2026-11-06).** Before the last cell writes anything, every output that already exists is compared with what the run made. A table counts as unchanged when its columns, shape and text are exactly equal and every number agrees to relative 1e-12 or absolute 1e-15; a figure only when its PNG is byte-identical. Unchanged outputs are not rewritten (they keep their checksum); any other difference stops the notebook before anything is written. Why: separate runs differ in the last bit of floating-point rounding (checked: the data's row order is the same in every process; the rerun with 4b/4c gave 27 byte-identical outputs and 3 tables differing by at most 2.2e-16). *Rejected:* byte-identity for tables — it fails on that rounding alone.

**Findings (preliminary)** are in the notebook's reading cells.
