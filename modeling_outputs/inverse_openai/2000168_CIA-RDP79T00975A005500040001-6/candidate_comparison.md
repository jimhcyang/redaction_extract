# Inverse Candidate Comparison: 2000168_CIA-RDP79T00975A005500040001-6

This file is meant for human inspection. It pairs the local answer-key text with the generated candidates for each redaction box.

## Source

- Cleaned text source: `postprocessed\2000168_CIA-RDP79T00975A005500040001-6\difference\unredacted_bracketed.filtered.aligned.txt`
- Source PDFs:
  - [CIA-RDP79T00975A005500040001-6.pdf](source_pdfs/CIA-RDP79T00975A005500040001-6.pdf)
  - [cib_02000168.pdf](source_pdfs/cib_02000168.pdf)

## BOX_001

- Source redaction id: `1`
- Target character count: `735`
- Token count: `118`

### Ground Truth

```text
I. SPECIAL ITEM

Laos: Formal approval of the Boun Oum government by parliamentary vote on 4 January will probably be viewed by the Communist bloc as illegal. Souvanna Phouma, who is still in Cambodia, had declared that he would formally step down as premier if the Boun Oum government were invested by constitutional procedure.

Airlift flights into Laos continued on 4 January with all ten Soviet IL-14s active. The five Soviet MI-4 helicopters are scheduled to arrive in Hanoi on 5 January.

Moscow statements dwell on alleged differences among Western allies over means of solving the crisis in Laos, while North Vietnam has issued a new appeal for another Geneva Conference and reactivation of the International Control Commission
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_001_CAND_01 | baseline_local_continuation | 160 | -575 | False | False |
| BOX_001_CAND_02 | alternate_actor_or_source | 244 | -491 | False | False |
| BOX_001_CAND_03 | escalation_or_heightened_risk | 294 | -441 | False | False |
| BOX_001_CAND_04 | deescalation_or_limited_impact | 312 | -423 | False | False |
| BOX_001_CAND_05 | diplomatic_or_policy_angle | 339 | -396 | False | False |
| BOX_001_CAND_06 | military_security_angle | 319 | -416 | False | False |
| BOX_001_CAND_07 | domestic_political_angle | 330 | -405 | False | False |
| BOX_001_CAND_08 | economic_logistical_angle | 300 | -435 | False | False |
| BOX_001_CAND_09 | uncertainty_or_source_dispute | 293 | -442 | False | False |
| BOX_001_CAND_10 | contrarian_document_level_hypothesis | 403 | -332 | False | False |

### Candidate Texts

#### BOX_001_CAND_01: baseline_local_continuation

- Actual chars: `160`
- Target chars: `735`
- Length delta: `-575`
- Rationale: Continues the immediately preceding sentence in the Turkey section.
- Distinctiveness: Local continuation that preserves the document’s explanatory tone.

```text
failed to attract popular support to the new regime. Some of the key ministers such as Foreign Minister Sarper will probably remain in the new cabinet. (Page 1)
```

#### BOX_001_CAND_02: alternate_actor_or_source

- Actual chars: `244`
- Target chars: `735`
- Length delta: `-491`
- Rationale: Shifts emphasis toward opposition actors shaping public opinion.
- Distinctiveness: Changes the main mechanism from minister selection to opposition mobilization.

```text
in large part because opposition leaders abroad and dissatisfied unions had been able to exploit the election gap. The convocation timetable may thus be influenced less by constitutional practice than by bargaining among party bosses in Ankara.
```

#### BOX_001_CAND_03: escalation_or_heightened_risk

- Actual chars: `294`
- Target chars: `735`
- Length delta: `-441`
- Rationale: Adds heightened instability tied to demonstrations and security strain.
- Distinctiveness: Introduces a risk-driven explanation involving unrest and army pressure.

```text
failed to prevent growing street demonstrations, and several provincial CNU organizers have reported that security forces are already strained by labor unrest. If the new Constituent Assembly is delayed, President Gursel may have to broaden the cabinet with figures acceptable to army elements.
```

#### BOX_001_CAND_04: deescalation_or_limited_impact

- Actual chars: `312`
- Target chars: `735`
- Length delta: `-423`
- Rationale: Reframes impact as limited and primarily reputational.
- Distinctiveness: Softens severity and treats the issue as communications/administrative performance.

```text
failed to attract popular support largely because technicians lacked visible accomplishments, but there is no evidence that this has threatened the regime’s basic stability. The new cabinet is expected to restore confidence mainly through administrative continuity and better communication with party supporters.
```

#### BOX_001_CAND_05: diplomatic_or_policy_angle

- Actual chars: `339`
- Target chars: `735`
- Length delta: `-396`
- Rationale: Connects ministerial choices to foreign-policy messaging.
- Distinctiveness: Shifts hypothesis toward external diplomacy and alliance reassurance.

```text
failed to attract popular support to the new regime. Some of the key ministers such as Foreign Minister Sarper will probably remain in the new cabinet. President Gursel may also seek to reassure Western and Balkan partners by retaining experienced foreign policy figures while presenting the reshuffle as a step toward political pluralism.
```

#### BOX_001_CAND_06: military_security_angle

- Actual chars: `319`
- Target chars: `735`
- Length delta: `-416`
- Rationale: Introduces military confidence as the driver of cabinet composition.
- Distinctiveness: Changes the key actor to the armed forces and a security-minded cabinet selection.

```text
failed because junior officers became convinced that civilian technicians would not reform internal discipline. As a result, President Gursel is likely to include at least one or two officers affiliated with the Committee of National Union to reassure the armed forces and prevent covert opposition within the barracks.
```

#### BOX_001_CAND_07: domestic_political_angle

- Actual chars: `330`
- Target chars: `735`
- Length delta: `-405`
- Rationale: Re-centers the explanation on party consolidation and patronage.
- Distinctiveness: Alters the mechanism from professional experience to party power and patronage.

```text
failed to attract popular support because the technicians’ program was perceived as overcentralized and insufficiently responsive to party demands. The new Constituent Assembly provides a vehicle for consolidating the CNU’s control by rewarding influential politicians and reducing the role of technocrats in day-to-day decisions.
```

#### BOX_001_CAND_08: economic_logistical_angle

- Actual chars: `300`
- Target chars: `735`
- Length delta: `-435`
- Rationale: Adds an economic delivery/logistics rationale for popularity.
- Distinctiveness: Focuses the cause on provincial economic performance and implementation logistics.

```text
failed to attract popular support because economic measures announced during the previous period were slow in reaching the provinces. President Gursel may select ministers with ties to finance and regional administration so that new budgets and public works can be presented promptly to the Assembly.
```

#### BOX_001_CAND_09: uncertainty_or_source_dispute

- Actual chars: `293`
- Target chars: `735`
- Length delta: `-442`
- Rationale: Frames the interpretation as uncertain and contested by observers.
- Distinctiveness: Emphasizes conflicting reporting and lack of clarity about depth of change.

```text
failed to attract popular support, although the extent is disputed among Ankara observers. Some press reports stress that the new cabinet will be no more than a change of personnel, while other reports suggest that Gursel intends a deeper political realignment within the Constituent Assembly.
```

#### BOX_001_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `403`
- Target chars: `735`
- Length delta: `-332`
- Rationale: Provides a document-level alternative: constitutional maneuver to constrain rivals.
- Distinctiveness: Shifts from popularity/technicians to constitutional/legal power-management as the real purpose.

```text
failed to attract popular support to the new regime. The resignation and convocation schedule appear to be primarily a constitutional maneuver designed to neutralize legal challenges rather than to improve governance. Gursel may use the reshuffle to limit the influence of rival factions within the Assembly and thereby reduce the likelihood of early amendments that could constrain executive authority.
```


## BOX_002

- Source redaction id: `2`
- Target character count: `416`
- Token count: `72`

### Ground Truth

```text
UAR (Syria)-Jordan: Syrian dissidents are still plotting to overthrow the UAR's regime in Syria and have again sought support, including weapons, from Jordan.
The conspiracy is said to include some of the high-ranking unit commanders in the Syrian Army and to have the support of most of the inhabitants of the Jabal Druze.
Amman has asked for more details and promised that aid will be provided if the plot is valid
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_002_CAND_01 | baseline_local_continuation | 495 | 79 | False | False |
| BOX_002_CAND_02 | alternate_actor_or_source | 309 | -107 | False | False |
| BOX_002_CAND_03 | escalation_or_heightened_risk | 278 | -138 | False | False |
| BOX_002_CAND_04 | deescalation_or_limited_impact | 288 | -128 | False | False |
| BOX_002_CAND_05 | diplomatic_or_policy_angle | 260 | -156 | False | False |
| BOX_002_CAND_06 | military_security_angle | 270 | -146 | False | False |
| BOX_002_CAND_07 | domestic_political_angle | 264 | -152 | False | False |
| BOX_002_CAND_08 | economic_logistical_angle | 270 | -146 | False | False |
| BOX_002_CAND_09 | uncertainty_or_source_dispute | 298 | -118 | False | False |
| BOX_002_CAND_10 | contrarian_document_level_hypothesis | 314 | -102 | False | False |

### Candidate Texts

#### BOX_002_CAND_01: baseline_local_continuation

- Actual chars: `495`
- Target chars: `416`
- Length delta: `79`
- Rationale: Keeps the Pakistan paragraph coherent as a local continuation.
- Distinctiveness: Direct continuation of the stated logic in the section.

```text
have called for "gradual disengagement" from Western alliances and for improved relations with the bloc. While President Ayub has made it clear that Pakistan remains committed to its alliance with the US and strongly supported Western positions during his recent Middle and Far Eastern tours, the government probably hopes such press comment will serve as a reminder to the United States that it should not take Pakistan's cooperation for granted and increase aid to India at Pakistan's expense.
```

#### BOX_002_CAND_02: alternate_actor_or_source

- Actual chars: `309`
- Target chars: `416`
- Length delta: `-107`
- Rationale: Shifts actor strategy toward negotiations and signaling.
- Distinctiveness: Changes the mechanism from “reminder” to bargaining and conditionality.

```text
advocate a tougher bargaining stance in which Pakistan will demand tangible concessions from Washington before honoring alliance obligations. The military government may be using editorials as a channel to signal to Indian and Soviet contacts that Pakistan can adjust its posture if aid terms are unfavorable.
```

#### BOX_002_CAND_03: escalation_or_heightened_risk

- Actual chars: `278`
- Target chars: `416`
- Length delta: `-138`
- Rationale: Introduces escalation to basing/overflight constraints.
- Distinctiveness: Adds concrete alliance-abrogation risk beyond editorial diplomacy.

```text
seek to prepare public opinion for a partial withdrawal from SEATO and for possible restrictions on US basing and overflight. If Western reactions are slow or negative, the military government could move from rhetoric to procedural steps that would complicate alliance planning.
```

#### BOX_002_CAND_04: deescalation_or_limited_impact

- Actual chars: `288`
- Target chars: `416`
- Length delta: `-128`
- Rationale: Reframes as rhetorical balancing with little treaty impact.
- Distinctiveness: Treats the episode as domestically motivated, low-impact foreign policy.

```text
stress only that Pakistan wants a more balanced posture while continuing practical cooperation with the United States in economic and security matters. The government probably believes that modest “disengagement” language can defuse domestic criticism without altering treaty commitments.
```

#### BOX_002_CAND_05: diplomatic_or_policy_angle

- Actual chars: `260`
- Target chars: `416`
- Length delta: `-156`
- Rationale: Focuses on diplomatic leverage and multilateral positioning.
- Distinctiveness: Shifts to UN/diplomatic bargaining outcomes rather than aid competition.

```text
aim to improve Pakistan’s diplomatic leverage by restoring contacts with the Soviet bloc while keeping the US informed. The military regime likely calculates that better east-west relations will strengthen Pakistan’s position in the UN and in talks with India.
```

#### BOX_002_CAND_06: military_security_angle

- Actual chars: `270`
- Target chars: `416`
- Length delta: `-146`
- Rationale: Adds a defense-supply and procurement angle to press comment.
- Distinctiveness: Changes the causal channel from political reminders to arms supply leverage.

```text
argue that Pakistan should diversify external supplies of arms and training and avoid becoming overly dependent on US deliveries. The government probably uses the editorials to keep Washington from complacency on Pakistan’s defense assistance and procurement priorities.
```

#### BOX_002_CAND_07: domestic_political_angle

- Actual chars: `264`
- Target chars: `416`
- Length delta: `-152`
- Rationale: Explains the editorials as faction-management within Pakistan.
- Distinctiveness: Shifts main purpose to domestic coalition stabilization.

```text
reflects internal pressures from nationalist elements within the military and from civilian critics demanding a more independent line. By elevating “gradual disengagement,” the regime can reconcile factions while maintaining Ayub’s personal commitment to the West.
```

#### BOX_002_CAND_08: economic_logistical_angle

- Actual chars: `270`
- Target chars: `416`
- Length delta: `-146`
- Rationale: Adds an economic/trade and credit logistics motive.
- Distinctiveness: Moves hypothesis toward trade diversion and development financing.

```text
prepare the public for efforts to secure trade credits and commodity arrangements from bloc countries. Pakistan’s leaders may believe that redirecting some economic transactions will offset the political costs of requesting additional Western assistance for development.
```

#### BOX_002_CAND_09: uncertainty_or_source_dispute

- Actual chars: `298`
- Target chars: `416`
- Length delta: `-118`
- Rationale: Frames as uncertain intent and possibly non-binding editorial posture.
- Distinctiveness: Emphasizes ambiguity of policy direction and timing.

```text
may be only partial indicators of official policy, since editorial positions in Pakistan have sometimes preceded real decisions by months. It is not clear whether the military government intends a substantive change or merely tolerates language aimed at influencing Washington and domestic opinion.
```

#### BOX_002_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `314`
- Target chars: `416`
- Length delta: `-102`
- Rationale: Contrarian: editorial strategy to manage India relations rather than US aid competition.
- Distinctiveness: Shifts expected outcome from US leverage/aid to Pakistan-India tension reduction.

```text
serve less as a warning to the US than as preparation for improved relations with India by reducing Pakistan’s incentive to remain aligned against New Delhi. The military regime may be testing whether a more open east-west posture can lower bilateral tensions while keeping security arrangements largely unchanged.
```


## BOX_003

- Source redaction id: `3`
- Target character count: `13`
- Token count: `3`

### Ground Truth

```text
Ne Win agrees
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_003_CAND_01 | baseline_local_continuation | 4 | -9 | False | False |
| BOX_003_CAND_02 | alternate_actor_or_source | 6 | -7 | False | False |
| BOX_003_CAND_03 | escalation_or_heightened_risk | 5 | -8 | False | False |
| BOX_003_CAND_04 | deescalation_or_limited_impact | 6 | -7 | False | False |
| BOX_003_CAND_05 | diplomatic_or_policy_angle | 5 | -8 | False | False |
| BOX_003_CAND_06 | military_security_angle | 7 | -6 | False | False |
| BOX_003_CAND_07 | domestic_political_angle | 7 | -6 | False | False |
| BOX_003_CAND_08 | economic_logistical_angle | 6 | -7 | False | False |
| BOX_003_CAND_09 | uncertainty_or_source_dispute | 7 | -6 | False | False |
| BOX_003_CAND_10 | contrarian_document_level_hypothesis | 6 | -7 | False | False |

### Candidate Texts

#### BOX_003_CAND_01: baseline_local_continuation

- Actual chars: `4`
- Target chars: `13`
- Length delta: `-9`
- Rationale: Minimal verb consistent with “asserts” context.
- Distinctiveness: Local grammatical fit only.

```text
adds
```

#### BOX_003_CAND_02: alternate_actor_or_source

- Actual chars: `6`
- Target chars: `13`
- Length delta: `-7`
- Rationale: Alternative verb matching the quoted intensity.
- Distinctiveness: Same local function but different reporting emphasis.

```text
claims
```

#### BOX_003_CAND_03: escalation_or_heightened_risk

- Actual chars: `5`
- Target chars: `13`
- Length delta: `-8`
- Rationale: Stronger pre-emptive tone for “serious proportions.”
- Distinctiveness: Shifts tone toward urgency.

```text
warns
```

#### BOX_003_CAND_04: deescalation_or_limited_impact

- Actual chars: `6`
- Target chars: `13`
- Length delta: `-7`
- Rationale: Neutral reporting tone.
- Distinctiveness: Reduces urgency implied by the verb.

```text
states
```

#### BOX_003_CAND_05: diplomatic_or_policy_angle

- Actual chars: `5`
- Target chars: `13`
- Length delta: `-8`
- Rationale: Less intense verb that still flows.
- Distinctiveness: Shifts from warning/assessment to noting a fact.

```text
notes
```

#### BOX_003_CAND_06: military_security_angle

- Actual chars: `7`
- Target chars: `13`
- Length delta: `-6`
- Rationale: Often used in intelligence writing before quotations.
- Distinctiveness: Emphasizes reporting channel.

```text
reports
```

#### BOX_003_CAND_07: domestic_political_angle

- Actual chars: `7`
- Target chars: `13`
- Length delta: `-6`
- Rationale: Fits with internal political leadership pressure.
- Distinctiveness: Adds political firmness.

```text
insists
```

#### BOX_003_CAND_08: economic_logistical_angle

- Actual chars: `6`
- Target chars: `13`
- Length delta: `-7`
- Rationale: Generic verb that can precede serious-proportions claims.
- Distinctiveness: Shifts to argumentation framing.

```text
argues
```

#### BOX_003_CAND_09: uncertainty_or_source_dispute

- Actual chars: `7`
- Target chars: `13`
- Length delta: `-6`
- Rationale: Introduces doubt/hedge consistent with intelligence reports.
- Distinctiveness: Adds uncertainty to the attribution.

```text
appears
```

#### BOX_003_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `6`
- Target chars: `13`
- Length delta: `-7`
- Rationale: Contrarian softer verb, implying interpretation rather than fact.
- Distinctiveness: Shifts to opinion rather than hard assertion.

```text
opines
```


## BOX_004

- Source redaction id: `4`
- Target character count: `21`
- Token count: `4`

### Ground Truth

```text
REPUBLIC OF THE CONGO
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_004_CAND_01 | baseline_local_continuation | 10 | -11 | False | False |
| BOX_004_CAND_02 | alternate_actor_or_source | 11 | -10 | False | False |
| BOX_004_CAND_03 | escalation_or_heightened_risk | 15 | -6 | False | False |
| BOX_004_CAND_04 | deescalation_or_limited_impact | 12 | -9 | False | False |
| BOX_004_CAND_05 | diplomatic_or_policy_angle | 16 | -5 | False | False |
| BOX_004_CAND_06 | military_security_angle | 19 | -2 | False | False |
| BOX_004_CAND_07 | domestic_political_angle | 18 | -3 | False | False |
| BOX_004_CAND_08 | economic_logistical_angle | 29 | 8 | False | False |
| BOX_004_CAND_09 | uncertainty_or_source_dispute | 18 | -3 | False | False |
| BOX_004_CAND_10 | contrarian_document_level_hypothesis | 19 | -2 | False | False |

### Candidate Texts

#### BOX_004_CAND_01: baseline_local_continuation

- Actual chars: `10`
- Target chars: `21`
- Length delta: `-11`
- Rationale: Fits phrase before “of the party for party leadership”.
- Distinctiveness: Local modifier continuing sentence rhythm.

```text
in effect,
```

#### BOX_004_CAND_02: alternate_actor_or_source

- Actual chars: `11`
- Target chars: `21`
- Length delta: `-10`
- Rationale: Common intelligence hedge.
- Distinctiveness: Shifts to source/attribution emphasis.

```text
reportedly,
```

#### BOX_004_CAND_03: escalation_or_heightened_risk

- Actual chars: `15`
- Target chars: `21`
- Length delta: `-6`
- Rationale: Adds intensity consistent with leadership threat.
- Distinctiveness: Inserts coercive risk mechanism.

```text
under pressure,
```

#### BOX_004_CAND_04: deescalation_or_limited_impact

- Actual chars: `12`
- Target chars: `21`
- Length delta: `-9`
- Rationale: Softens implication of immediate loss of control.
- Distinctiveness: Reduces severity.

```text
if possible,
```

#### BOX_004_CAND_05: diplomatic_or_policy_angle

- Actual chars: `16`
- Target chars: `21`
- Length delta: `-5`
- Rationale: Suggests political bargain.
- Distinctiveness: Reframes as negotiation tactic.

```text
as a compromise,
```

#### BOX_004_CAND_06: military_security_angle

- Actual chars: `19`
- Target chars: `21`
- Length delta: `-2`
- Rationale: Implies coordination with military actors.
- Distinctiveness: Adds security-related coordination.

```text
after consultation,
```

#### BOX_004_CAND_07: domestic_political_angle

- Actual chars: `18`
- Target chars: `21`
- Length delta: `-3`
- Rationale: Places conflict inside party politics.
- Distinctiveness: Changes actor focus to internal opponents.

```text
with party rivals,
```

#### BOX_004_CAND_08: economic_logistical_angle

- Actual chars: `29`
- Target chars: `21`
- Length delta: `8`
- Rationale: Administrative/logistical procedural link.
- Distinctiveness: Shifts mechanism to bureaucracy/timing.

```text
pending administrative steps,
```

#### BOX_004_CAND_09: uncertainty_or_source_dispute

- Actual chars: `18`
- Target chars: `21`
- Length delta: `-3`
- Rationale: Hedge consistent with intelligence uncertainty.
- Distinctiveness: Introduces contested interpretation.

```text
according to some,
```

#### BOX_004_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `19`
- Target chars: `21`
- Length delta: `-2`
- Rationale: Contrarian tone suggesting earlier promises fail.
- Distinctiveness: Shifts document-level narrative toward backsliding.

```text
despite assurances,
```


## BOX_005

- Source redaction id: `5`
- Target character count: `460`
- Token count: `74`

### Ground Truth

```text
Congo: an IL-14 aircraft which landed at Gemena on 1 January, ostensibly carrying spare parts and welfare supplies for the UAR battalion in Equateur Province, included money, arms, and presumably technical personnel for the Gizenga dissidents. Gemena, about 400 miles northwest of Stanleyville, is the location of the UAR's Congo battalion.
the UN "is not to be notified" of the names of persons aboard the plane, who were to be described merely as technicians
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_005_CAND_01 | baseline_local_continuation | 183 | -277 | False | False |
| BOX_005_CAND_02 | alternate_actor_or_source | 195 | -265 | False | False |
| BOX_005_CAND_03 | escalation_or_heightened_risk | 212 | -248 | False | False |
| BOX_005_CAND_04 | deescalation_or_limited_impact | 191 | -269 | False | False |
| BOX_005_CAND_05 | diplomatic_or_policy_angle | 219 | -241 | False | False |
| BOX_005_CAND_06 | military_security_angle | 248 | -212 | False | False |
| BOX_005_CAND_07 | domestic_political_angle | 223 | -237 | False | False |
| BOX_005_CAND_08 | economic_logistical_angle | 175 | -285 | False | False |
| BOX_005_CAND_09 | uncertainty_or_source_dispute | 210 | -250 | False | False |
| BOX_005_CAND_10 | contrarian_document_level_hypothesis | 251 | -209 | False | False |

### Candidate Texts

#### BOX_005_CAND_01: baseline_local_continuation

- Actual chars: `183`
- Target chars: `460`
- Length delta: `-277`
- Rationale: Matches the preceding France-Algeria sentence fragment structure.
- Distinctiveness: Local continuation with same causal chain.

```text
lowered army morale since the government's decision to hold the referendum on its Algerian policy and since the pro-rebel Moslem demonstrations during De Gaulle's 9-13 December visit.
```

#### BOX_005_CAND_02: alternate_actor_or_source

- Actual chars: `195`
- Target chars: `460`
- Length delta: `-265`
- Rationale: Shifts cause to manpower/conscription grievances.
- Distinctiveness: Changes causal driver from referendum/rebel demonstrations to personnel hardships.

```text
has been driven more by resentment over conscription hardships than by the referendum itself, and junior officers in Algeria have complained that promised rotations and leave have been cancelled.
```

#### BOX_005_CAND_03: escalation_or_heightened_risk

- Actual chars: `212`
- Target chars: `460`
- Length delta: `-248`
- Rationale: Adds serious disruptive military outcomes.
- Distinctiveness: Heightens risk from morale to sabotage/desertion and potential refusal.

```text
will likely culminate in sabotage of supply convoys and coordinated desertions among units along the Moroccan border, with some officers suggesting the army may refuse orders if the referendum result is negative.
```

#### BOX_005_CAND_04: deescalation_or_limited_impact

- Actual chars: `191`
- Target chars: `460`
- Length delta: `-269`
- Rationale: Limits impact geographically/organizationally.
- Distinctiveness: Reduces severity by attributing morale effects to limited units.

```text
is affecting attitudes only in certain garrisons, while most senior commanders appear to accept De Gaulle's political timetable and focus on maintaining order pending instructions from Paris.
```

#### BOX_005_CAND_05: diplomatic_or_policy_angle

- Actual chars: `219`
- Target chars: `460`
- Length delta: `-241`
- Rationale: Links morale to fears about diplomacy/negotiations.
- Distinctiveness: Policy-mechanism angle: diplomatic settlement concerns.

```text
reflects unease that De Gaulle's decision signals a move toward negotiated settlement with rebels, which officers fear will undermine France's influence in North Africa and reduce bargaining leverage with NATO partners.
```

#### BOX_005_CAND_06: military_security_angle

- Actual chars: `248`
- Target chars: `460`
- Length delta: `-212`
- Rationale: Security contingency planning and uprising rumors.
- Distinctiveness: Introduces operational security planning and force posture changes.

```text
has increased concern that Algerian urban security will deteriorate after voting, since rumor of an uprising is already spreading through officers' channels and some commanders are quietly drawing contingency plans for deployment of reserve troops.
```

#### BOX_005_CAND_07: domestic_political_angle

- Actual chars: `223`
- Target chars: `460`
- Length delta: `-237`
- Rationale: Domestic French politics consequence.
- Distinctiveness: Connects Algerian morale to metropolitan parliamentary pressure.

```text
suggests that political factions within the French army are using Algeria to judge the credibility of civilian leadership at home, and this may accelerate calls in metropolitan France for parliamentary censure of De Gaulle.
```

#### BOX_005_CAND_08: economic_logistical_angle

- Actual chars: `175`
- Target chars: `460`
- Length delta: `-285`
- Rationale: Economic/sector shortages as driver.
- Distinctiveness: Logistical readiness and procurement shortages, not political events.

```text
is also tied to shortages of pay and equipment for units in Algeria, and officers believe that any delay in procurement will impair readiness when demonstrations turn violent.
```

#### BOX_005_CAND_09: uncertainty_or_source_dispute

- Actual chars: `210`
- Target chars: `460`
- Length delta: `-250`
- Rationale: Explicit uncertainty about breadth of dissatisfaction.
- Distinctiveness: Emphasizes conflicting reporting and limited scope.

```text
appears to have weakened morale, but reports differ on how widespread the dissatisfaction is; some sources believe criticism is largely confined to a minority of officers strongly influenced by the June letter.
```

#### BOX_005_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `251`
- Target chars: `460`
- Length delta: `-209`
- Rationale: Contrarian: compliance motives over ideological opposition.
- Distinctiveness: Shifts the interpretation toward self-interest/personal protection rather than policy opposition.

```text
does not necessarily indicate opposition to De Gaulle, since many officers may be preparing to comply with orders regardless of the referendum outcome; their concern may be more about protecting their own positions than about changing national policy.
```


## BOX_006

- Source redaction id: `6`
- Target character count: `56`
- Token count: `11`

### Ground Truth

```text
Minister of State for Algeria Louis Joxe is said to have
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_006_CAND_01 | baseline_local_continuation | 32 | -24 | False | False |
| BOX_006_CAND_02 | alternate_actor_or_source | 31 | -25 | False | False |
| BOX_006_CAND_03 | escalation_or_heightened_risk | 29 | -27 | False | False |
| BOX_006_CAND_04 | deescalation_or_limited_impact | 32 | -24 | False | False |
| BOX_006_CAND_05 | diplomatic_or_policy_angle | 31 | -25 | False | False |
| BOX_006_CAND_06 | military_security_angle | 29 | -27 | False | False |
| BOX_006_CAND_07 | domestic_political_angle | 29 | -27 | False | False |
| BOX_006_CAND_08 | economic_logistical_angle | 28 | -28 | False | False |
| BOX_006_CAND_09 | uncertainty_or_source_dispute | 38 | -18 | False | False |
| BOX_006_CAND_10 | contrarian_document_level_hypothesis | 33 | -23 | False | False |

### Candidate Texts

#### BOX_006_CAND_01: baseline_local_continuation

- Actual chars: `32`
- Target chars: `56`
- Length delta: `-24`
- Rationale: Directly fits the quoted continuation in the Algeria section.
- Distinctiveness: Local continuity with reported speech.

```text
commented that "serious trouble"
```

#### BOX_006_CAND_02: alternate_actor_or_source

- Actual chars: `31`
- Target chars: `56`
- Length delta: `-25`
- Rationale: Alternative reporting verb still grammatically compatible.
- Distinctiveness: Changes source framing via verb.

```text
declared that "serious trouble"
```

#### BOX_006_CAND_03: escalation_or_heightened_risk

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: More urgent tone before the quote.
- Distinctiveness: Inserts urgency/heightened risk framing.

```text
warned that "serious trouble"
```

#### BOX_006_CAND_04: deescalation_or_limited_impact

- Actual chars: `32`
- Target chars: `56`
- Length delta: `-24`
- Rationale: Softer verb implying possibility rather than certainty.
- Distinctiveness: Reduces certainty level.

```text
suggested that "serious trouble"
```

#### BOX_006_CAND_05: diplomatic_or_policy_angle

- Actual chars: `31`
- Target chars: `56`
- Length delta: `-25`
- Rationale: Neutral verb; still works with policy attitudes.
- Distinctiveness: More analytical/observational than warning.

```text
observed that "serious trouble"
```

#### BOX_006_CAND_06: military_security_angle

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: Straight statement fits intelligence tone.
- Distinctiveness: Emphasizes factual assertion.

```text
stated that "serious trouble"
```

#### BOX_006_CAND_07: domestic_political_angle

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: Adds argumentative nuance.
- Distinctiveness: Shifts to political persuasion rather than mere report.

```text
argued that "serious trouble"
```

#### BOX_006_CAND_08: economic_logistical_angle

- Actual chars: `28`
- Target chars: `56`
- Length delta: `-28`
- Rationale: Neutral note consistent with intelligence writing.
- Distinctiveness: Slightly less severe emphasis than warning.

```text
noted that "serious trouble"
```

#### BOX_006_CAND_09: uncertainty_or_source_dispute

- Actual chars: `38`
- Target chars: `56`
- Length delta: `-18`
- Rationale: Adds attribution/hedge using “reportedly”.
- Distinctiveness: Introduces source uncertainty.

```text
reportedly said that "serious trouble"
```

#### BOX_006_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `33`
- Target chars: `56`
- Length delta: `-23`
- Rationale: Implies insistence, potentially contestable.
- Distinctiveness: Suggests ongoing argument among officers.

```text
maintained that "serious trouble"
```


## BOX_007

- Source redaction id: `7`
- Target character count: `362`
- Token count: `50`

### Ground Truth

```text
Ecuador-USSR: Ecuadorean Foreign Minister Jose Chiriboga
"Yesterday I had the opportunity of discussing with the Russian ambassador future visits of a commercial nature." Chiriboga came to the United States in late December to discuss US economic assistance to Ecuador. Ecuador during the last month re-established active diplomatic relations with Czechoslovakia
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_007_CAND_01 | baseline_local_continuation | 184 | -178 | False | False |
| BOX_007_CAND_02 | alternate_actor_or_source | 181 | -181 | False | False |
| BOX_007_CAND_03 | escalation_or_heightened_risk | 184 | -178 | False | False |
| BOX_007_CAND_04 | deescalation_or_limited_impact | 182 | -180 | False | False |
| BOX_007_CAND_05 | diplomatic_or_policy_angle | 190 | -172 | False | False |
| BOX_007_CAND_06 | military_security_angle | 196 | -166 | False | False |
| BOX_007_CAND_07 | domestic_political_angle | 200 | -162 | False | False |
| BOX_007_CAND_08 | economic_logistical_angle | 186 | -176 | False | False |
| BOX_007_CAND_09 | uncertainty_or_source_dispute | 192 | -170 | False | False |
| BOX_007_CAND_10 | contrarian_document_level_hypothesis | 177 | -185 | False | False |

### Candidate Texts

#### BOX_007_CAND_01: baseline_local_continuation

- Actual chars: `184`
- Target chars: `362`
- Length delta: `-178`
- Rationale: Keeps the France-Algeria paragraph’s flow including general strike reference.
- Distinctiveness: Local continuation matching the immediate aftermath.

```text
lies ahead in Algeria and there will probably be a major uprising just before or during the referendum, both Moslems and Europeans in Oran are planning a general strike today. (Page 7)
```

#### BOX_007_CAND_02: alternate_actor_or_source

- Actual chars: `181`
- Target chars: `362`
- Length delta: `-181`
- Rationale: Shifts geography and implies clandestine organization linkage.
- Distinctiveness: Changes focus from timing to spread and covert infiltration.

```text
is expected to spread rapidly from Oran to other coastal cities, and commanders believe it may be linked to clandestine rebel organizers already embedding in European neighborhoods.
```

#### BOX_007_CAND_03: escalation_or_heightened_risk

- Actual chars: `184`
- Target chars: `362`
- Length delta: `-178`
- Rationale: Heightens risk to communications loss and street battles.
- Distinctiveness: Severity escalates beyond uprising/strike to tactical collapse.

```text
will probably culminate in a collapse of public order in Oran, with officers anticipating street battles and the temporary loss of control over key communications before voting closes.
```

#### BOX_007_CAND_04: deescalation_or_limited_impact

- Actual chars: `182`
- Target chars: `362`
- Length delta: `-180`
- Rationale: Downplays likelihood of major systemic unrest.
- Distinctiveness: Limits impact and adds containment expectation.

```text
may remain largely confined to demonstrations and a limited strike, and the army’s leadership expects it can restore control after police reinforcements arrive from nearby garrisons.
```

#### BOX_007_CAND_05: diplomatic_or_policy_angle

- Actual chars: `190`
- Target chars: `362`
- Length delta: `-172`
- Rationale: Connects unrest to political bargaining and policy reversal pressure.
- Distinctiveness: Policy implication rather than battlefield dynamics.

```text
could be used by De Gaulle’s opponents as evidence that negotiations with the rebel government are unavoidable, thereby increasing pressure on Paris to reconsider its post-referendum stance.
```

#### BOX_007_CAND_06: military_security_angle

- Actual chars: `196`
- Target chars: `362`
- Length delta: `-166`
- Rationale: Adds specific security posture actions.
- Distinctiveness: Force posture and movement controls as the key mechanism.

```text
could prompt a shift in security posture, since Algeria command may prepare reserve mobilization and impose movement restrictions to prevent rebel-led coordination before and after the referendum.
```

#### BOX_007_CAND_07: domestic_political_angle

- Actual chars: `200`
- Target chars: `362`
- Length delta: `-162`
- Rationale: Links Algeria events to French domestic political debate.
- Distinctiveness: Expected outcome focuses on metropolitan legitimacy and debate.

```text
will intensify public debate in France over De Gaulle’s leadership, and military spokesmen fear that the referendum outcome will be interpreted as a mandate for or against army involvement in Algeria.
```

#### BOX_007_CAND_08: economic_logistical_angle

- Actual chars: `186`
- Target chars: `362`
- Length delta: `-176`
- Rationale: Adds logistics/economic disruption through port and deliveries.
- Distinctiveness: Shifts mechanism to economic/logistics effects.

```text
will seriously disrupt trade and shipping through Oran’s port facilities, since dockworkers are likely to join the strike and disrupt fuel deliveries and provisioning for garrison units.
```

#### BOX_007_CAND_09: uncertainty_or_source_dispute

- Actual chars: `192`
- Target chars: `362`
- Length delta: `-170`
- Rationale: Introduces uncertainty about coordination and timing.
- Distinctiveness: Uncertainty axis: competing assessments of feasibility.

```text
may or may not materialize on schedule, because some officers question whether rebel organizers can coordinate the strike and uprising simultaneously, though they concede tensions remain high.
```

#### BOX_007_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `177`
- Target chars: `362`
- Length delta: `-185`
- Rationale: Contrarian: unrest could be negotiated bargaining rather than rebellion escalation.
- Distinctiveness: Document-level reinterpretation toward tactical bargaining.

```text
may be exaggerated by alarmists within the officer corps, and the strike in Oran could be a bargaining tactic aimed at forcing concessions without triggering a broader uprising.
```


## BOX_008

- Source redaction id: `8`
- Target character count: `71`
- Token count: `9`

### Ground Truth

```text
who lacked both political appeal and political comprehension. President
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_008_CAND_01 | baseline_local_continuation | 69 | -2 | False | False |
| BOX_008_CAND_02 | alternate_actor_or_source | 88 | 17 | False | False |
| BOX_008_CAND_03 | escalation_or_heightened_risk | 95 | 24 | False | False |
| BOX_008_CAND_04 | deescalation_or_limited_impact | 90 | 19 | False | False |
| BOX_008_CAND_05 | diplomatic_or_policy_angle | 97 | 26 | False | False |
| BOX_008_CAND_06 | military_security_angle | 75 | 4 | False | False |
| BOX_008_CAND_07 | domestic_political_angle | 83 | 12 | False | False |
| BOX_008_CAND_08 | economic_logistical_angle | 90 | 19 | False | False |
| BOX_008_CAND_09 | uncertainty_or_source_dispute | 80 | 9 | False | False |
| BOX_008_CAND_10 | contrarian_document_level_hypothesis | 82 | 11 | False | False |

### Candidate Texts

#### BOX_008_CAND_01: baseline_local_continuation

- Actual chars: `69`
- Target chars: `71`
- Length delta: `-2`
- Rationale: Connects directly to the Turkish cabinet resignation narrative.
- Distinctiveness: Local continuation bridging to Gursel’s aims.

```text
technicians. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_02: alternate_actor_or_source

- Actual chars: `88`
- Target chars: `71`
- Length delta: `17`
- Rationale: Shifts composition: provincial politicians rather than technicians.
- Distinctiveness: Changes actor pool/selection basis.

```text
politicians from the provinces. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_03: escalation_or_heightened_risk

- Actual chars: `95`
- Target chars: `71`
- Length delta: `24`
- Rationale: Introduces risk that new appointees may be suspect.
- Distinctiveness: Adds heightened security concern about appointees.

```text
men with tenuous security credentials. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_04: deescalation_or_limited_impact

- Actual chars: `90`
- Target chars: `71`
- Length delta: `19`
- Rationale: Frames new appointees as restrained rather than disruptive.
- Distinctiveness: Reduces perceived threat by limiting power.

```text
moderates with limited authority. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_05: diplomatic_or_policy_angle

- Actual chars: `97`
- Target chars: `71`
- Length delta: `26`
- Rationale: Focuses on image management for external audiences.
- Distinctiveness: Policy/diplomatic angle rather than internal technicality.

```text
figures acceptable to foreign observers. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_06: military_security_angle

- Actual chars: `75`
- Target chars: `71`
- Length delta: `4`
- Rationale: Introduces military affiliation in cabinet selection.
- Distinctiveness: Changes main actor class to military-linked figures.

```text
officers on leave. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_07: domestic_political_angle

- Actual chars: `83`
- Target chars: `71`
- Length delta: `12`
- Rationale: Shifts to domestic factional loyalty.
- Distinctiveness: Political alignment emphasized over professional background.

```text
hard-line party loyalists. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_08: economic_logistical_angle

- Actual chars: `90`
- Target chars: `71`
- Length delta: `19`
- Rationale: Economic administration expertise emphasized.
- Distinctiveness: Changes the selection criterion to fiscal/budget management.

```text
administrators noted for budgets. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_09: uncertainty_or_source_dispute

- Actual chars: `80`
- Target chars: `71`
- Length delta: `9`
- Rationale: Hedges selection characterization.
- Distinctiveness: Highlights uncertainty about the personnel makeup.

```text
as some rumors suggest. Gursel may also hope to promote his own political future
```

#### BOX_008_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `82`
- Target chars: `71`
- Length delta: `11`
- Rationale: Contrarian: nominal technicians but actually political operators.
- Distinctiveness: Document-level reinterpretation of cabinet personnel labeling.

```text
technicians in name only. Gursel may also hope to promote his own political future
```


## BOX_009

- Source redaction id: `9`
- Target character count: `117`
- Token count: `21`

### Ground Truth

```text
Ne Win told his colleagues that if the lack of government discipline continued, "the army would have to act very soon
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_009_CAND_01 | baseline_local_continuation | 64 | -53 | False | False |
| BOX_009_CAND_02 | alternate_actor_or_source | 94 | -23 | False | False |
| BOX_009_CAND_03 | escalation_or_heightened_risk | 125 | 8 | False | False |
| BOX_009_CAND_04 | deescalation_or_limited_impact | 102 | -15 | False | False |
| BOX_009_CAND_05 | diplomatic_or_policy_angle | 150 | 33 | False | False |
| BOX_009_CAND_06 | military_security_angle | 145 | 28 | False | False |
| BOX_009_CAND_07 | domestic_political_angle | 125 | 8 | False | False |
| BOX_009_CAND_08 | economic_logistical_angle | 169 | 52 | False | False |
| BOX_009_CAND_09 | uncertainty_or_source_dispute | 121 | 4 | False | False |
| BOX_009_CAND_10 | contrarian_document_level_hypothesis | 148 | 31 | False | False |

### Candidate Texts

#### BOX_009_CAND_01: baseline_local_continuation

- Actual chars: `64`
- Target chars: `117`
- Length delta: `-53`
- Rationale: Fits the surrounding sentence fragment about takeover timing.
- Distinctiveness: Local continuation with the takeover quote.

```text
suggests that "an army takeover may occur in February or March."
```

#### BOX_009_CAND_02: alternate_actor_or_source

- Actual chars: `94`
- Target chars: `117`
- Length delta: `-23`
- Rationale: Adds an alternate source/channel (Rangoon contacts).
- Distinctiveness: Changes intelligence sourcing while keeping meaning.

```text
reports from Rangoon contacts indicate that "an army takeover may occur in February or March."
```

#### BOX_009_CAND_03: escalation_or_heightened_risk

- Actual chars: `125`
- Target chars: `117`
- Length delta: `8`
- Rationale: Heightens urgency and links to deterioration.
- Distinctiveness: Adds conditional escalation and forcing mechanism.

```text
warns that "an army takeover may occur in February or March" and that Ne Win may be forced to act sooner if disorder worsens.
```

#### BOX_009_CAND_04: deescalation_or_limited_impact

- Actual chars: `102`
- Target chars: `117`
- Length delta: `-15`
- Rationale: Adds conditionality that reduces likelihood.
- Distinctiveness: Lower-impact scenario dependent on negotiations.

```text
states that "an army takeover may occur in February or March" but only if political negotiations fail.
```

#### BOX_009_CAND_05: diplomatic_or_policy_angle

- Actual chars: `150`
- Target chars: `117`
- Length delta: `33`
- Rationale: Introduces foreign-observer/diplomatic pressure angle.
- Distinctiveness: Shifts causal explanation to external pressure.

```text
notes that foreign observers believe "an army takeover may occur in February or March" as a result of regional pressures and concerns about stability.
```

#### BOX_009_CAND_06: military_security_angle

- Actual chars: `145`
- Target chars: `117`
- Length delta: `28`
- Rationale: Military-readiness reasoning tied to arms control.
- Distinctiveness: Mechanism becomes security management over politics.

```text
assesses that "an army takeover may occur in February or March" because commanders want to secure key garrisons and arms stocks before elections.
```

#### BOX_009_CAND_07: domestic_political_angle

- Actual chars: `125`
- Target chars: `117`
- Length delta: `8`
- Rationale: Links takeover timing to party dispute.
- Distinctiveness: Changes causal driver to internal party conflict.

```text
indicates that "an army takeover may occur in February or March" as a maneuver to resolve the Union party leadership dispute.
```

#### BOX_009_CAND_08: economic_logistical_angle

- Actual chars: `169`
- Target chars: `117`
- Length delta: `52`
- Rationale: Economic deterioration as the driver.
- Distinctiveness: Shifts to governance-capacity/economic collapse explanation.

```text
concludes that "an army takeover may occur in February or March" because economic collapse and administrative deterioration leave the army as the only effective manager.
```

#### BOX_009_CAND_09: uncertainty_or_source_dispute

- Actual chars: `121`
- Target chars: `117`
- Length delta: `4`
- Rationale: Emphasizes incomplete evidence and timing risk.
- Distinctiveness: Adds uncertainty about the date and certainty level.

```text
cautions that "an army takeover may occur in February or March" though evidence remains incomplete and timing could slip.
```

#### BOX_009_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `148`
- Target chars: `117`
- Length delta: `31`
- Rationale: Contrarian: threat as intimidation, not actual plan.
- Distinctiveness: Reinterprets intent behind takeover talk.

```text
suggests that "an army takeover may occur in February or March" may be a tactic to intimidate civilian leaders rather than a planned coup by Ne Win.
```


## BOX_010

- Source redaction id: `10`
- Target character count: `295`
- Token count: `50`

### Ground Truth

```text
He told his staff and brigade commanders that he had hoped the present leaders would "step up the pace of managing the government, but they were too busy fighting among themselves," and he reportedly added, "The Burmese people have again shown that they are not ready for democracy and self-rule
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_010_CAND_01 | baseline_local_continuation | 242 | -53 | False | False |
| BOX_010_CAND_02 | alternate_actor_or_source | 216 | -79 | False | False |
| BOX_010_CAND_03 | escalation_or_heightened_risk | 198 | -97 | False | False |
| BOX_010_CAND_04 | deescalation_or_limited_impact | 216 | -79 | False | False |
| BOX_010_CAND_05 | diplomatic_or_policy_angle | 207 | -88 | False | False |
| BOX_010_CAND_06 | military_security_angle | 204 | -91 | False | False |
| BOX_010_CAND_07 | domestic_political_angle | 189 | -106 | False | False |
| BOX_010_CAND_08 | economic_logistical_angle | 224 | -71 | False | False |
| BOX_010_CAND_09 | uncertainty_or_source_dispute | 204 | -91 | False | False |
| BOX_010_CAND_10 | contrarian_document_level_hypothesis | 218 | -77 | False | False |

### Candidate Texts

#### BOX_010_CAND_01: baseline_local_continuation

- Actual chars: `242`
- Target chars: `295`
- Length delta: `-53`
- Rationale: Continues the Ne Win resumption logic and adds officers’ pressure consistent with the paragraph.
- Distinctiveness: Local continuation plus added supportive detail.

```text
are convincing Ne Win that he should resume office. Although some reports suggest he may compromise with Nu, his senior officers are pressing for a direct return to army administration and predict that civilian authority cannot restore order.
```

#### BOX_010_CAND_02: alternate_actor_or_source

- Actual chars: `216`
- Target chars: `295`
- Length delta: `-79`
- Rationale: Changes source of influence to army intelligence channel.
- Distinctiveness: Shifts causal weight to intelligence assessment.

```text
have strengthened Ne Win’s position but his decision is not yet final, and the most persuasive guidance may be coming from senior army intelligence rather than from the general administrative failures cited publicly.
```

#### BOX_010_CAND_03: escalation_or_heightened_risk

- Actual chars: `198`
- Target chars: `295`
- Length delta: `-97`
- Rationale: Adds escalation: armed clashes prospect.
- Distinctiveness: Turns administrative deterioration into imminent violence risk.

```text
are convincing Ne Win that the elected government can no longer prevent rapid expansion of unrest, and he may be forced to resume office by the prospect of armed clashes in the towns by late winter.
```

#### BOX_010_CAND_04: deescalation_or_limited_impact

- Actual chars: `216`
- Target chars: `295`
- Length delta: `-79`
- Rationale: Deescalates to incremental reforms rather than full resume.
- Distinctiveness: Alternative mechanism: stronger deputy/coordination.

```text
are convincing Ne Win that corrective measures are needed, but he may seek to do so without a formal resumption of the premiership by appointing a stronger deputy and tightening coordination with civilian ministries.
```

#### BOX_010_CAND_05: diplomatic_or_policy_angle

- Actual chars: `207`
- Target chars: `295`
- Length delta: `-88`
- Rationale: Adds policy/diplomatic motive: restoring foreign confidence.
- Distinctiveness: Shifts from internal order to external economic reassurance.

```text
are convincing Ne Win that foreign confidence in Burma’s stability is declining, and his return to office could be used to reassure overseas creditors and encourage external assistance for economic recovery.
```

#### BOX_010_CAND_06: military_security_angle

- Actual chars: `204`
- Target chars: `295`
- Length delta: `-91`
- Rationale: Security logistics route protection as driver.
- Distinctiveness: Moves mechanism to supply-route security.

```text
are convincing Ne Win that lawlessness is undermining army control of key supply routes, and he may resume office to impose security discipline and protect border logistics against insurgent interference.
```

#### BOX_010_CAND_07: domestic_political_angle

- Actual chars: `189`
- Target chars: `295`
- Length delta: `-106`
- Rationale: Links decision to preventing party/command fragmentation.
- Distinctiveness: Causal driver becomes intra-elite competition.

```text
are convincing Ne Win that he must resume office to prevent the Union party from splitting further and weakening his own influence, which he believes would allow rival commanders to emerge.
```

#### BOX_010_CAND_08: economic_logistical_angle

- Actual chars: `224`
- Target chars: `295`
- Length delta: `-71`
- Rationale: Economic services and food distribution motive.
- Distinctiveness: Focuses on rationing and basic services logistics.

```text
are convincing Ne Win that the deterioration of the economy has reached the point where civilian administrators cannot secure food distribution and basic services, leaving the army the only organization capable of rationing.
```

#### BOX_010_CAND_09: uncertainty_or_source_dispute

- Actual chars: `204`
- Target chars: `295`
- Length delta: `-91`
- Rationale: Highlights dependency and divided reporting.
- Distinctiveness: Uncertainty axis: timing conditional on Nu’s stabilization.

```text
are convincing Ne Win in principle, but the timing depends on whether Nu can stabilize the law-and-order situation for long enough to preserve political legitimacy; reports remain divided on likely dates.
```

#### BOX_010_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `218`
- Target chars: `295`
- Length delta: `-77`
- Rationale: Contrarian: personal standing protection motive.
- Distinctiveness: Shifts from governance necessity to personal/political survival.

```text
are convincing Ne Win that resignation from the prime ministership has already damaged his standing, and he may return primarily to protect his personal role in the army rather than to reverse specific economic trends.
```


## BOX_011

- Source redaction id: `11`
- Target character count: `75`
- Token count: `12`

### Ground Truth

```text
Nu considered the party poorly organized, with many "wrong" people in power
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_011_CAND_01 | baseline_local_continuation | 31 | -44 | False | False |
| BOX_011_CAND_02 | alternate_actor_or_source | 53 | -22 | False | False |
| BOX_011_CAND_03 | escalation_or_heightened_risk | 60 | -15 | False | False |
| BOX_011_CAND_04 | deescalation_or_limited_impact | 52 | -23 | False | False |
| BOX_011_CAND_05 | diplomatic_or_policy_angle | 68 | -7 | False | False |
| BOX_011_CAND_06 | military_security_angle | 62 | -13 | False | False |
| BOX_011_CAND_07 | domestic_political_angle | 59 | -16 | False | False |
| BOX_011_CAND_08 | economic_logistical_angle | 65 | -10 | False | False |
| BOX_011_CAND_09 | uncertainty_or_source_dispute | 54 | -21 | False | False |
| BOX_011_CAND_10 | contrarian_document_level_hypothesis | 88 | 13 | False | False |

### Candidate Texts

#### BOX_011_CAND_01: baseline_local_continuation

- Actual chars: `31`
- Target chars: `75`
- Length delta: `-44`
- Rationale: Fits as a phrase describing dissension and leading to takeover risk.
- Distinctiveness: Local descriptive tone.

```text
suggests persistent dissension,
```

#### BOX_011_CAND_02: alternate_actor_or_source

- Actual chars: `53`
- Target chars: `75`
- Length delta: `-22`
- Rationale: Changes the implied group showing dissension.
- Distinctiveness: Actor focus shifts from party to regional commanders.

```text
indicates that some regional commanders are wavering,
```

#### BOX_011_CAND_03: escalation_or_heightened_risk

- Actual chars: `60`
- Target chars: `75`
- Length delta: `-15`
- Rationale: More serious than “persistent” dissension.
- Distinctiveness: Severity increases to open confrontation.

```text
reports that open confrontation within the party is growing,
```

#### BOX_011_CAND_04: deescalation_or_limited_impact

- Actual chars: `52`
- Target chars: `75`
- Length delta: `-23`
- Rationale: Less severe, not yet public.
- Distinctiveness: Reduces visibility/impact.

```text
notes that the dissension has not yet become public,
```

#### BOX_011_CAND_05: diplomatic_or_policy_angle

- Actual chars: `68`
- Target chars: `75`
- Length delta: `-7`
- Rationale: Policy legitimacy angle with civilian groups.
- Distinctiveness: Shifts toward credibility with civilian supporters.

```text
shows that Nu’s line is losing credibility with key civilian groups,
```

#### BOX_011_CAND_06: military_security_angle

- Actual chars: `62`
- Target chars: `75`
- Length delta: `-13`
- Rationale: Security/army advisory disagreements.
- Distinctiveness: Mechanism becomes army-adviser factionalism.

```text
points to disagreements among army advisers to the government,
```

#### BOX_011_CAND_07: domestic_political_angle

- Actual chars: `59`
- Target chars: `75`
- Length delta: `-16`
- Rationale: Domestic party rivalry framed directly.
- Distinctiveness: Changes internal political map.

```text
reflects rivalry between Nu’s faction and party executives,
```

#### BOX_011_CAND_08: economic_logistical_angle

- Actual chars: `65`
- Target chars: `75`
- Length delta: `-10`
- Rationale: Economic distribution/patronage disputes.
- Distinctiveness: Causal driver becomes resources distribution.

```text
stems from disputes over patronage and distribution of resources,
```

#### BOX_011_CAND_09: uncertainty_or_source_dispute

- Actual chars: `54`
- Target chars: `75`
- Length delta: `-21`
- Rationale: Hedges verification status.
- Distinctiveness: Emphasizes inability to verify.

```text
cannot be fully verified but appears to be increasing,
```

#### BOX_011_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `88`
- Target chars: `75`
- Length delta: `13`
- Rationale: Contrarian: dissension as bargaining/negotiated transition.
- Distinctiveness: Reinterprets political struggle as controlled negotiation setup.

```text
may be less about policy and more about preparations for a negotiated transfer of power,
```


## BOX_012

- Source redaction id: `12`
- Target character count: `16`
- Token count: `3`

### Ground Truth

```text
a political hack
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_012_CAND_01 | baseline_local_continuation | 17 | 1 | False | False |
| BOX_012_CAND_02 | alternate_actor_or_source | 13 | -3 | False | False |
| BOX_012_CAND_03 | escalation_or_heightened_risk | 9 | -7 | False | False |
| BOX_012_CAND_04 | deescalation_or_limited_impact | 8 | -8 | False | False |
| BOX_012_CAND_05 | diplomatic_or_policy_angle | 12 | -4 | False | False |
| BOX_012_CAND_06 | military_security_angle | 15 | -1 | False | False |
| BOX_012_CAND_07 | domestic_political_angle | 15 | -1 | False | False |
| BOX_012_CAND_08 | economic_logistical_angle | 19 | 3 | False | False |
| BOX_012_CAND_09 | uncertainty_or_source_dispute | 10 | -6 | False | False |
| BOX_012_CAND_10 | contrarian_document_level_hypothesis | 14 | -2 | False | False |

### Candidate Texts

#### BOX_012_CAND_01: baseline_local_continuation

- Actual chars: `17`
- Target chars: `16`
- Length delta: `1`
- Rationale: Matches the nearby phrase about Kyaw Dun and party role.
- Distinctiveness: Local continuity with title usage.

```text
Secretary General
```

#### BOX_012_CAND_02: alternate_actor_or_source

- Actual chars: `13`
- Target chars: `16`
- Length delta: `-3`
- Rationale: Alternative title consistent with internal party roles.
- Distinctiveness: Changes the implied authority level of Kyaw Dun.

```text
Deputy leader
```

#### BOX_012_CAND_03: escalation_or_heightened_risk

- Actual chars: `9`
- Target chars: `16`
- Length delta: `-7`
- Rationale: Adds sharper characterization of Kyaw Dun.
- Distinctiveness: Shifts toward severity/ideological posture.

```text
hardliner
```

#### BOX_012_CAND_04: deescalation_or_limited_impact

- Actual chars: `8`
- Target chars: `16`
- Length delta: `-8`
- Rationale: Softer characterization.
- Distinctiveness: Reduces perceived threat from Kyaw Dun.

```text
moderate
```

#### BOX_012_CAND_05: diplomatic_or_policy_angle

- Actual chars: `12`
- Target chars: `16`
- Length delta: `-4`
- Rationale: Policy-focused framing.
- Distinctiveness: Shifts from office/title to policy role framing.

```text
policy chief
```

#### BOX_012_CAND_06: military_security_angle

- Actual chars: `15`
- Target chars: `16`
- Length delta: `-1`
- Rationale: Security liaison framing though within party context.
- Distinctiveness: Introduces security-connection hypothesis.

```text
liaison officer
```

#### BOX_012_CAND_07: domestic_political_angle

- Actual chars: `15`
- Target chars: `16`
- Length delta: `-1`
- Rationale: Domestic organizational framing.
- Distinctiveness: Changes role to organizer/power broker.

```text
party organizer
```

#### BOX_012_CAND_08: economic_logistical_angle

- Actual chars: `19`
- Target chars: `16`
- Length delta: `3`
- Rationale: Administrative framing consistent with intra-party functions.
- Distinctiveness: Shifts to bureaucratic management hypothesis.

```text
administrative head
```

#### BOX_012_CAND_09: uncertainty_or_source_dispute

- Actual chars: `10`
- Target chars: `16`
- Length delta: `-6`
- Rationale: Hedge to indicate uncertainty about party title.
- Distinctiveness: Changes certainty/attribution rather than substantive role.

```text
reportedly
```

#### BOX_012_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `14`
- Target chars: `16`
- Length delta: `-2`
- Rationale: Contrarian title implying broader leadership conflict.
- Distinctiveness: Alters document-level implication about the scale of contest over leadership.

```text
rival chairman
```


## BOX_013

- Source redaction id: `13`
- Target character count: `56`
- Token count: `11`

### Ground Truth

```text
Minister of State for Algeria Louis Joxe is said to have
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_013_CAND_01 | baseline_local_continuation | 32 | -24 | False | False |
| BOX_013_CAND_02 | alternate_actor_or_source | 29 | -27 | False | False |
| BOX_013_CAND_03 | escalation_or_heightened_risk | 29 | -27 | False | False |
| BOX_013_CAND_04 | deescalation_or_limited_impact | 32 | -24 | False | False |
| BOX_013_CAND_05 | diplomatic_or_policy_angle | 28 | -28 | False | False |
| BOX_013_CAND_06 | military_security_angle | 29 | -27 | False | False |
| BOX_013_CAND_07 | domestic_political_angle | 29 | -27 | False | False |
| BOX_013_CAND_08 | economic_logistical_angle | 31 | -25 | False | False |
| BOX_013_CAND_09 | uncertainty_or_source_dispute | 38 | -18 | False | False |
| BOX_013_CAND_10 | contrarian_document_level_hypothesis | 31 | -25 | False | False |

### Candidate Texts

#### BOX_013_CAND_01: baseline_local_continuation

- Actual chars: `32`
- Target chars: `56`
- Length delta: `-24`
- Rationale: Direct fit to the quoted phrase in the De Gaulle/Algeria section.
- Distinctiveness: Local continuation with reported speech.

```text
commented that "serious trouble"
```

#### BOX_013_CAND_02: alternate_actor_or_source

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: Alternative verb with similar meaning.
- Distinctiveness: Changes reporting tone and attribution style.

```text
opined that "serious trouble"
```

#### BOX_013_CAND_03: escalation_or_heightened_risk

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: More assertive than “commented”.
- Distinctiveness: Raises certainty about trouble ahead.

```text
stated that "serious trouble"
```

#### BOX_013_CAND_04: deescalation_or_limited_impact

- Actual chars: `32`
- Target chars: `56`
- Length delta: `-24`
- Rationale: Softened implication.
- Distinctiveness: Reduces severity/certainty.

```text
suggested that "serious trouble"
```

#### BOX_013_CAND_05: diplomatic_or_policy_angle

- Actual chars: `28`
- Target chars: `56`
- Length delta: `-28`
- Rationale: Neutral analytical tone.
- Distinctiveness: Frames as observation for policy analysis.

```text
noted that "serious trouble"
```

#### BOX_013_CAND_06: military_security_angle

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: Security-mobilization connotation.
- Distinctiveness: Introduces a warning/security posture.

```text
warned that "serious trouble"
```

#### BOX_013_CAND_07: domestic_political_angle

- Actual chars: `29`
- Target chars: `56`
- Length delta: `-27`
- Rationale: Argumentative tone consistent with officer debates.
- Distinctiveness: Shifts from reporting to persuasion among officers.

```text
argued that "serious trouble"
```

#### BOX_013_CAND_08: economic_logistical_angle

- Actual chars: `31`
- Target chars: `56`
- Length delta: `-25`
- Rationale: Neutral verb.
- Distinctiveness: Slightly shifts toward analytical appraisal.

```text
observed that "serious trouble"
```

#### BOX_013_CAND_09: uncertainty_or_source_dispute

- Actual chars: `38`
- Target chars: `56`
- Length delta: `-18`
- Rationale: Adds hedge via reportedly.
- Distinctiveness: Emphasizes source uncertainty.

```text
reportedly said that "serious trouble"
```

#### BOX_013_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `31`
- Target chars: `56`
- Length delta: `-25`
- Rationale: Implies insistence; could reflect disputed claim.
- Distinctiveness: Contrarian: suggests contentious insistence rather than consensus.

```text
insisted that "serious trouble"
```
