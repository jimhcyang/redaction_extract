# Inverse Candidate Comparison: 2000171_CIA-RDP79T00975A005500080001-2

This file is meant for human inspection. It pairs the local answer-key text with the generated candidates for each redaction box.

## Source

- Cleaned text source: `postprocessed\2000171_CIA-RDP79T00975A005500080001-2\difference\unredacted_bracketed.filtered.aligned.txt`
- Source PDFs:
  - [CIA-RDP79T00975A005500080001-2.pdf](source_pdfs/CIA-RDP79T00975A005500080001-2.pdf)
  - [cib_02000171.pdf](source_pdfs/cib_02000171.pdf)

## BOX_001

- Source redaction id: `1`
- Target character count: `21`
- Token count: `4`

### Ground Truth

```text
I. THE COMMUNIST BLOC
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_001_CAND_01 | baseline_local_continuation | 16 | -5 | False | False |
| BOX_001_CAND_02 | alternate_actor_or_source | 26 | 5 | False | False |
| BOX_001_CAND_03 | escalation_or_heightened_risk | 24 | 3 | False | False |
| BOX_001_CAND_04 | deescalation_or_limited_impact | 28 | 7 | False | False |
| BOX_001_CAND_05 | diplomatic_or_policy_angle | 18 | -3 | False | False |
| BOX_001_CAND_06 | military_security_angle | 26 | 5 | False | False |
| BOX_001_CAND_07 | domestic_political_angle | 32 | 11 | False | False |
| BOX_001_CAND_08 | economic_logistical_angle | 28 | 7 | False | False |
| BOX_001_CAND_09 | uncertainty_or_source_dispute | 30 | 9 | False | False |
| BOX_001_CAND_10 | contrarian_document_level_hypothesis | 25 | 4 | False | False |

### Candidate Texts

#### BOX_001_CAND_01: baseline_local_continuation

- Actual chars: `16`
- Target chars: `21`
- Length delta: `-5`
- Rationale: A short DAILY BRIEF label introducing the Communist China item that follows immediately.
- Distinctiveness: Keeps the same actor and category as the next sentence.

```text
Communist China.
```

#### BOX_001_CAND_02: alternate_actor_or_source

- Actual chars: `26`
- Target chars: `21`
- Length delta: `5`
- Rationale: Could be an alternative subheading on the same DAILY BRIEF list, shifting focus to Asia-Africa.
- Distinctiveness: Changes the main geopolitical focus from China to Laos.

```text
Laos: political situation.
```

#### BOX_001_CAND_03: escalation_or_heightened_risk

- Actual chars: `24`
- Target chars: `21`
- Length delta: `3`
- Rationale: A compact escalation-style teaser matching the bulletin’s later Cuba/Guantanamo content.
- Distinctiveness: Shifts to a higher-stakes security flashpoint (Cuba/Guantanamo).

```text
Guantanamo crisis warns.
```

#### BOX_001_CAND_04: deescalation_or_limited_impact

- Actual chars: `28`
- Target chars: `21`
- Length delta: `7`
- Rationale: A generic de-escalatory phrase consistent with intelligence reporting tone.
- Distinctiveness: Alters severity down and implies uncertainty/limited scope.

```text
Developments remain limited.
```

#### BOX_001_CAND_05: diplomatic_or_policy_angle

- Actual chars: `18`
- Target chars: `21`
- Length delta: `-3`
- Rationale: Would fit the WEST section’s multilateral/diplomatic theme seen later in the bulletin.
- Distinctiveness: Shifts from Communist China to a policy/diplomacy frame.

```text
OAS talks on Cuba.
```

#### BOX_001_CAND_06: military_security_angle

- Actual chars: `26`
- Target chars: `21`
- Length delta: `5`
- Rationale: Compact militarized teaser consistent with the Katanga/UN tension noted on the next page.
- Distinctiveness: Shifts to military-security posture in Congo.

```text
Katanga: UN frontier alert
```

#### BOX_001_CAND_07: domestic_political_angle

- Actual chars: `32`
- Target chars: `21`
- Length delta: `11`
- Rationale: A domestic-politics teaser reflecting popular discontent theme.
- Distinctiveness: Keeps China but frames it as internal political legitimacy risk rather than aid/foreign ties.

```text
Riot danger spreads inside China
```

#### BOX_001_CAND_08: economic_logistical_angle

- Actual chars: `28`
- Target chars: `21`
- Length delta: `7`
- Rationale: Economic scarcity as the cause, matching the detailed China food-shortage paragraph later.
- Distinctiveness: Centers on economic/logistical causation rather than generic discontent.

```text
Food shortages drive unrest.
```

#### BOX_001_CAND_09: uncertainty_or_source_dispute

- Actual chars: `30`
- Target chars: `21`
- Length delta: `9`
- Rationale: Matches the bulletin’s style for uncertain/uncorroborated items.
- Distinctiveness: Shifts the hypothesis to source uncertainty rather than a concrete event.

```text
Unconfirmed reports circulate.
```

#### BOX_001_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `25`
- Target chars: `21`
- Length delta: `4`
- Rationale: A document-level contrarian angle: suggesting discontent as a bloc-provocation mechanism.
- Distinctiveness: Reframes the cause as deliberate bloc influence rather than purely local shortages.

```text
Bloc pressure via unrest.
```


## BOX_002

- Source redaction id: `2`
- Target character count: `1375`
- Token count: `219`

### Ground Truth

```text
II. ASIA-AFRICA

Laos: The Communist airlift into Laos continues. Nine flights, possibly to the Vang Vieng area, were confirmed on 9 January; eleven flights to Vang Vieng are scheduled on 10 January.

Four T-6 aircraft were scheduled to arrive in Savannakhet on 9 January, and are to be flown to Vientiane on 10 January. The T-6s will give the Laotians a capability of interdicting the Soviet airlift. Supplies for transshipment to Laos are probably being moved into North Vietnam by rail.

the North Vietnamese refuse to permit the ICC to inspect a train possibly transporting military equipment from Communist China on 23 December. This train was at Lao Kay, the North Vietnamese entry point on the rail line from Kunming to Hanoi. The North Vietnamese denied the inspection on the grounds that the train was a “local,” allegedly arriving from another part of North Vietnam.

Congo: Indications that the Gizenga dissidents are continuing to extend their control of areas of the eastern Congo have coincided with reports of uncoordinated countermeasures on the part of the Mobutu regime. An emissary of Mobutu is in Elisabethville for talks with Katanga President Tshombe concerning the possibility of Katangan financial support for Mobutu’s forces. In Leopoldville, however, Mobutu’s commissioner for finance reportedly assured UN representative Dayal on 5 January that the
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_002_CAND_01 | baseline_local_continuation | 764 | -611 | False | False |
| BOX_002_CAND_02 | alternate_actor_or_source | 915 | -460 | False | False |
| BOX_002_CAND_03 | escalation_or_heightened_risk | 947 | -428 | False | False |
| BOX_002_CAND_04 | deescalation_or_limited_impact | 828 | -547 | False | False |
| BOX_002_CAND_05 | diplomatic_or_policy_angle | 1009 | -366 | False | False |
| BOX_002_CAND_06 | military_security_angle | 951 | -424 | False | False |
| BOX_002_CAND_07 | domestic_political_angle | 990 | -385 | False | False |
| BOX_002_CAND_08 | economic_logistical_angle | 980 | -395 | False | False |
| BOX_002_CAND_09 | uncertainty_or_source_dispute | 1023 | -352 | False | False |
| BOX_002_CAND_10 | contrarian_document_level_hypothesis | 1094 | -281 | False | False |

### Candidate Texts

#### BOX_002_CAND_01: baseline_local_continuation

- Actual chars: `764`
- Target chars: `1375`
- Length delta: `-611`
- Rationale: Continues the Africa/Congo thread immediately following the Burma aid section, in a manner consistent with the bulletin’s operational tone.
- Distinctiveness: Presumes the missing material is about UN logistics, reconnaissance, and assessment of rebel movements.

```text
Leopoldville reports that the UN has resumed air reconnaissance over the Bukavu-Kamina corridor to assess rebel troop movements. UN officials believe that Gizenga’s supporters are attempting to secure a bridgehead in eastern Kasai to cut the main supply route to Elisabethville. At the same time, the UN is considering whether to extend its neutral-zone arrangements to the transport lines along the Lualaba. The Secretary-General has asked the parties for additional evidence on the size and composition of the forces entering Katanga and has warned that further violations would compel stronger action. Meanwhile, in Stanleyville, Lumumbist authorities continue to press for recognition and to demand that the UN remove contingents from positions deemed hostile.
```

#### BOX_002_CAND_02: alternate_actor_or_source

- Actual chars: `915`
- Target chars: `1375`
- Length delta: `-460`
- Rationale: Introduces a different intelligence source (Canadian UN observers) and a negotiated/elite channel focus rather than purely UN threat management.
- Distinctiveness: Shifts the actor emphasis toward Tshombe-side bargaining and elite intermediaries.

```text
Congo contacts attributed to Canadian UN observers indicate that Tshombe’s own commanders have been quietly negotiating with Lumumbist intermediaries to prevent a broader Katanga clash. According to these reports, the UN has received feelers suggesting that some units loyal to Tshombe would withdraw from contested crossings in return for guarantees of amnesty and release of certain captured officials. Leopoldville, however, rejects any compromise that would preserve a separate Katanga administration and insists that the only acceptable settlement would be complete integration under a central authority. The prospects for an agreement are described as marginal, with both sides apparently using negotiations to buy time for redeployment. In Washington, diplomatic channels note that European governments have urged restraint to avoid giving new momentum to Soviet propaganda about “neo-colonial” interference.
```

#### BOX_002_CAND_03: escalation_or_heightened_risk

- Actual chars: `947`
- Target chars: `1375`
- Length delta: `-428`
- Rationale: Makes the situation more urgent by adding sabotage, fuel depots, and a likely broader retaliation cycle.
- Distinctiveness: Heightens risk and adds new mechanisms (sabotage, fuel disruption) beyond the local setup.

```text
UN estimates have grown more pessimistic as additional Lumumbist columns are reported to be moving toward the key river crossings near Kalemie and Kindu. There are indications that some elements are preparing to sabotage communications and to seize fuel depots required for UN aircraft operations. Tshombe is believed to be pressing for rapid redeployment of loyalist forces to occupy strategic posts along the Katanga-Léopoldville approaches, while publicly insisting that any UN restraint is an unacceptable concession. Leopoldville warns that if Tshombe orders an occupation of UN-controlled neutral zones, the Congolese central government will treat it as a direct challenge to international authority and will retaliate by cutting off logistical support for Tshombe’s external backers. The Secretary-General’s latest message characterizes the situation as deteriorating and calls for immediate disarmament verification by an expanded UN team.
```

#### BOX_002_CAND_04: deescalation_or_limited_impact

- Actual chars: `828`
- Target chars: `1375`
- Length delta: `-547`
- Rationale: Frames missing content as uncertainty and limited impact, emphasizing verification and restraint.
- Distinctiveness: Lowers severity and focuses on containable border policing rather than an imminent major offensive.

```text
Recent UN liaison contacts suggest that the most recent troop movements in Katanga may be limited to small-scale border policing rather than preparation for a major offensive. Officials note that several reported “invaders” could be regular units temporarily relocated to control bandit activity and protect local communications. UN commanders are reluctant to expand neutral-zone boundaries until they can verify the exact identity and mission of the forces concerned. Leopoldville officials indicate that they may tolerate a temporary suspension of hostilities provided that normal traffic along the main routes is restored. The Secretary-General expects that continued talks, rather than force, will resolve the immediate dispute and that any economic measures, including blockades, will be constrained to specific incidents.
```

#### BOX_002_CAND_05: diplomatic_or_policy_angle

- Actual chars: `1009`
- Target chars: `1375`
- Length delta: `-366`
- Rationale: Shifts the missing material toward policy/diplomatic bargaining and sanction design rather than only troop posture.
- Distinctiveness: Introduces a multi-step cease-fire and sanctions compliance framework as the core of the passage.

```text
In New York, UN political officers report that negotiations are being shaped around a phased diplomatic package: first, an agreement on cease-fire monitoring procedures; second, establishment of a joint timetable for disarmament; and third, a review of Katanga’s administrative status within the Congolese framework. Several nonaligned states have urged that any sanctions imposed on parties to the Katanga dispute be tied to verifiable compliance benchmarks to avoid encouraging further resistance. Diplomatic sources indicate that the Secretary-General is attempting to keep the issue insulated from wider East-West contestation by soliciting written assurances from key governments regarding restraint on military shipments. Tshombe is said to prefer symbolic measures that preserve his bargaining position, while Lumumbist representatives press for recognition of their authority in Stanleyville as a condition for participation. The resulting policy tradeoffs are described as difficult but not hopeless.
```

#### BOX_002_CAND_06: military_security_angle

- Actual chars: `951`
- Target chars: `1375`
- Length delta: `-424`
- Rationale: Adds concrete security/collection elements: air corridors, signals, maritime team, and surveillance needs.
- Distinctiveness: Centers on force posture and intelligence/monitoring rather than negotiations or economics.

```text
UN commanders have requested additional landing slots for medical evacuations at Elisabethville and a temporary rerouting of transport aircraft to avoid hostile small-arms zones reported near the Katanga frontier. Satellite and signals monitoring cited in UN briefings suggests that communications traffic has increased along several routes used for moving men and ammunition. There is also concern that fast boats could be used to smuggle supplies across the Congo River during periods of poor visibility. The Secretary-General’s staff is therefore considering the deployment of a small maritime security team and tighter controls on fuel shipments to both UN and local units. Leopoldville insists that any security posture must be linked to disarmament verification, while Tshombe argues that restraints should not limit his ability to protect UN corridors. The overall security picture is assessed as volatile and requiring continuous surveillance.
```

#### BOX_002_CAND_07: domestic_political_angle

- Actual chars: `990`
- Target chars: `1375`
- Length delta: `-385`
- Rationale: Introduces internal politics and propaganda pressure driving potential escalatory behavior.
- Distinctiveness: Treats the missing passage as domestic legitimacy and propaganda dynamics rather than military logistics.

```text
Political maneuvering around Katanga is intensifying as local officials seek to solidify internal support for their stance toward the UN. In Elisabethville, pro-Tshombe leaders have been urging public demonstrations to demonstrate unity and to counter criticism from moderates who warn that prolonged confrontation will weaken the economy. Lumumbist authorities, facing pressure from towns threatened by disruptions, are stepping up propaganda calling for popular defense and demanding that all “traitors” be removed from civil administration. In Leopoldville, central government figures are attempting to balance the need to show toughness against Katanga with the risk of alienating potential allies among federalist elements. Diplomats report that rumors of high-level defections are being used by both sides to pressure the other party. The result is a heightened political sensitivity that could lead to escalatory actions by local commanders acting to satisfy domestic constituencies.
```

#### BOX_002_CAND_08: economic_logistical_angle

- Actual chars: `980`
- Target chars: `1375`
- Length delta: `-395`
- Rationale: Focuses on economic and logistics pressures (food/fuel shortages, procurement, customs revenue).
- Distinctiveness: Reframes the driver of the situation as economic/logistical strain pushing parties to negotiations.

```text
Economic officials in the UN system are warning that transport interruptions are quickly translating into shortages of food and fuel in eastern provinces. Reports from trading centers suggest that merchants are hoarding supplies in anticipation of renewed fighting, which in turn raises prices and undermines UN procurement plans. Leopoldville’s finance authorities claim that the disruption of customs revenue will impair the central government’s ability to pay civil salaries and to maintain basic administrative services. UN staff therefore plan to relax certain procurement rules to permit emergency purchases, but only if parties provide guarantees of safe corridors for trucks. There are also indications that rebel-controlled areas are diverting industrial outputs and attempting to redirect shipments through alternative routes to avoid blockade measures. These logistical pressures are considered one of the principal reasons the parties may be forced toward negotiation.
```

#### BOX_002_CAND_09: uncertainty_or_source_dispute

- Actual chars: `1023`
- Target chars: `1375`
- Length delta: `-352`
- Rationale: Presents conflicting assessments and delayed verification, consistent with intelligence uncertainty language.
- Distinctiveness: Makes the passage centrally about disagreement over facts and conditional policy.

```text
Conflicting reports from field observers make it difficult to determine the true scope of Lumumbist forces moving into Katanga. Some UN-linked observers state that only a small detachment has entered the province and that most other movement consists of routine reassignments, while other sources claim that a larger invasion is underway. Tshombe’s representatives insist that the UN neutral zones are already being targeted by armed bands acting with Lumumbist direction, but Leopoldville disputes that characterization and attributes the activity to local security problems. The UN Secretariat has therefore delayed a definitive assessment pending identification of uniforms, command structures, and the chain of supply used by the entering units. Because of the uncertainty, staff discussions lean toward cautious language in any public communique so as not to prejudice later verification. As a result, the near-term policy response is described as conditional and subject to revision once further evidence is obtained.
```

#### BOX_002_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `1094`
- Target chars: `1375`
- Length delta: `-281`
- Rationale: Provides a document-level contrarian strategic framing: propaganda/UN-credibility manipulation leading to stalemate.
- Distinctiveness: Shifts the causal mechanism to a broader geopolitical strategy aimed at undermining UN authority.

```text
Analysts consider that the immediate Katanga flare-up may be serving a broader strategic purpose unrelated to local control alone. It could be intended to demonstrate that the UN cannot sustain effective authority in Africa, thereby encouraging bloc-aligned actors to argue that international guarantees are unreliable. Under this hypothesis, the parties may be tolerating limited violence to elicit predictable reactions—such as economic measures or expanded UN deployments—that can then be portrayed as external interference. The selection of particular targets, including communications sites and transport corridors, would be aimed less at winning territory outright than at provoking a sustained diplomatic crisis. If this assessment is correct, the most likely outcome would be an extended stalemate punctuated by intermittent skirmishes, with both sides waiting for external political decisions rather than seeking a rapid settlement. The UN’s next steps are therefore expected to be incremental and highly constrained by the need to avoid actions that could be exploited for propaganda.
```


## BOX_003

- Source redaction id: `3`
- Target character count: `33`
- Token count: `5`

### Ground Truth

```text
An unconfirmed report states that
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_003_CAND_01 | baseline_local_continuation | 24 | -9 | False | False |
| BOX_003_CAND_02 | alternate_actor_or_source | 36 | 3 | False | False |
| BOX_003_CAND_03 | escalation_or_heightened_risk | 35 | 2 | False | False |
| BOX_003_CAND_04 | deescalation_or_limited_impact | 35 | 2 | False | False |
| BOX_003_CAND_05 | diplomatic_or_policy_angle | 42 | 9 | False | False |
| BOX_003_CAND_06 | military_security_angle | 34 | 1 | False | False |
| BOX_003_CAND_07 | domestic_political_angle | 41 | 8 | False | False |
| BOX_003_CAND_08 | economic_logistical_angle | 38 | 5 | False | False |
| BOX_003_CAND_09 | uncertainty_or_source_dispute | 33 | 0 | True | False |
| BOX_003_CAND_10 | contrarian_document_level_hypothesis | 42 | 9 | False | False |

### Candidate Texts

#### BOX_003_CAND_01: baseline_local_continuation

- Actual chars: `24`
- Target chars: `33`
- Length delta: `-9`
- Rationale: Fits as a lead-in to the statement that riots occurred in Harbin, keeping chronology intact.
- Distinctiveness: Keeps discontent/riots causation and sequential timeline.

```text
 Later, on the mainland,
```

#### BOX_003_CAND_02: alternate_actor_or_source

- Actual chars: `36`
- Target chars: `33`
- Length delta: `3`
- Rationale: Could reflect competing reporting about where riots began, changing confidence/source framing.
- Distinctiveness: Shifts from direct report to counterreports/dispute.

```text
 Meanwhile, counterreports suggested
```

#### BOX_003_CAND_03: escalation_or_heightened_risk

- Actual chars: `35`
- Target chars: `33`
- Length delta: `2`
- Rationale: Escalatory phrasing consistent with the subsequent arrest and execution figure.
- Distinctiveness: Emphasizes escalation in violence rather than simple occurrence.

```text
 By mid-December, violence worsened
```

#### BOX_003_CAND_04: deescalation_or_limited_impact

- Actual chars: `35`
- Target chars: `33`
- Length delta: `2`
- Rationale: A de-escalatory reading contrasting with the explicit riot description that follows.
- Distinctiveness: Opposes severity and implies containment.

```text
 Subsequently, unrest was contained
```

#### BOX_003_CAND_05: diplomatic_or_policy_angle

- Actual chars: `42`
- Target chars: `33`
- Length delta: `9`
- Rationale: Would frame the follow-on outcome as policy response, though it must match this exact spot.
- Distinctiveness: Shifts to regime policy actions after the slogans/riots.

```text
 The authorities then tightened discipline
```

#### BOX_003_CAND_06: military_security_angle

- Actual chars: `34`
- Target chars: `33`
- Length delta: `1`
- Rationale: Security-force action fits the pattern of arrests and execution mentioned nearby.
- Distinctiveness: Re-anchors the missing span to military/security measures.

```text
 Troops were sent to restore order
```

#### BOX_003_CAND_07: domestic_political_angle

- Actual chars: `41`
- Target chars: `33`
- Length delta: `8`
- Rationale: Directly ties to the following line about dissatisfaction among civilians in Dairen over army treatment.
- Distinctiveness: Frames internal civil-military tensions as the key mechanism.

```text
 The army’s favoritism drew further anger
```

#### BOX_003_CAND_08: economic_logistical_angle

- Actual chars: `38`
- Target chars: `33`
- Length delta: `5`
- Rationale: Connects to undernourishment and labor inability earlier in the page’s narrative.
- Distinctiveness: Centers the cause in logistics/nutrition rather than politics.

```text
 Under-nourishment continued to spread
```

#### BOX_003_CAND_09: uncertainty_or_source_dispute

- Actual chars: `33`
- Target chars: `33`
- Length delta: `0`
- Rationale: Could reflect uncertainty about magnitude before giving an arrest figure.
- Distinctiveness: Introduces uncertainty about size/extent rather than stating it definitively.

```text
 It was not clear how many rioted
```

#### BOX_003_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `42`
- Target chars: `33`
- Length delta: `9`
- Rationale: A contrarian mechanism: authorities attribute unrest to corruption rather than food scarcity alone.
- Distinctiveness: Recasts the narrative cause and political framing.

```text
 The riots were blamed on local corruption
```


## BOX_004

- Source redaction id: `4`
- Target character count: `200`
- Token count: `35`

### Ground Truth

```text
of the people in Havana in a state of frightened expectancy.

the US break “had a considerable impact upon the general public,” which had already begun to feel isolated from the rest of the hemisphere
```

### Candidate Summary

| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |
|---|---|---:|---:|---|---|
| BOX_004_CAND_01 | baseline_local_continuation | 143 | -57 | False | False |
| BOX_004_CAND_02 | alternate_actor_or_source | 170 | -30 | False | False |
| BOX_004_CAND_03 | escalation_or_heightened_risk | 151 | -49 | False | False |
| BOX_004_CAND_04 | deescalation_or_limited_impact | 146 | -54 | False | False |
| BOX_004_CAND_05 | diplomatic_or_policy_angle | 164 | -36 | False | False |
| BOX_004_CAND_06 | military_security_angle | 142 | -58 | False | False |
| BOX_004_CAND_07 | domestic_political_angle | 148 | -52 | False | False |
| BOX_004_CAND_08 | economic_logistical_angle | 139 | -61 | False | False |
| BOX_004_CAND_09 | uncertainty_or_source_dispute | 161 | -39 | False | False |
| BOX_004_CAND_10 | contrarian_document_level_hypothesis | 167 | -33 | False | False |

### Candidate Texts

#### BOX_004_CAND_01: baseline_local_continuation

- Actual chars: `143`
- Target chars: `200`
- Length delta: `-57`
- Rationale: Continues the Latin America section’s discussion of domestic reaction to the Cuba break, matching the trailing phrase “leaving the majority”.
- Distinctiveness: Local continuation that stays within pro-Castro domestic pressures.

```text
pro-Castro elements in the country would probably press the government for immediate concessions to Cuba and for leniency toward demonstrators.
```

#### BOX_004_CAND_02: alternate_actor_or_source

- Actual chars: `170`
- Target chars: `200`
- Length delta: `-30`
- Rationale: Shifts who drives domestic reaction (moderates in Congress) and links to coordination with Washington.
- Distinctiveness: Changes the internal political driver from pro-Castro pressure to moderate faction strategy.

```text
some of the more moderate factions are expected to exploit the crisis to strengthen their position in Congress and seek a controlled response coordinated with Washington.
```

#### BOX_004_CAND_03: escalation_or_heightened_risk

- Actual chars: `151`
- Target chars: `200`
- Length delta: `-49`
- Rationale: Raises severity beyond “relatively ineffective” demonstrations, aligning with the idea of tightened police controls.
- Distinctiveness: Adds escalation to clashes and harsh policing as likely outcomes.

```text
could lead to renewed street clashes and arrests, thereby increasing the likelihood that governments will overreact with harsher police-state measures.
```

#### BOX_004_CAND_04: deescalation_or_limited_impact

- Actual chars: `146`
- Target chars: `200`
- Length delta: `-54`
- Rationale: Downshifts impact, consistent with some regimes being reluctant yet not breaking decisively.
- Distinctiveness: Frames the outcome as limited operational impact rather than unrest.

```text
may remain largely verbal, with limited demonstrations that do not materially affect the governments’ ability to pursue cautious foreign policies.
```

#### BOX_004_CAND_05: diplomatic_or_policy_angle

- Actual chars: `164`
- Target chars: `200`
- Length delta: `-36`
- Rationale: Connects to the surrounding discussion of possible conferences and diplomatic management.
- Distinctiveness: Makes the core effect a shift toward diplomacy/conference strategy.

```text
will likely translate into calls for a special inter-American foreign ministers’ session designed to contain the conflict through negotiation rather than sanctions.
```

#### BOX_004_CAND_06: military_security_angle

- Actual chars: `142`
- Target chars: `200`
- Length delta: `-58`
- Rationale: Adds a security-operations response consistent with intelligence-style reporting on police state controls.
- Distinctiveness: Reframes domestic reaction into border/port security and surveillance.

```text
prompt security services to increase surveillance of ports and key border routes, anticipating attempts to move personnel or supplies to Cuba.
```

#### BOX_004_CAND_07: domestic_political_angle

- Actual chars: `148`
- Target chars: `200`
- Length delta: `-52`
- Rationale: Focuses on legitimacy and coalition stability, matching the bulletin’s election-cycle sensitivities elsewhere.
- Distinctiveness: Shifts from street demonstrations to internal legitimacy and coalition cohesion.

```text
will sharpen factional disputes over legitimacy, since ruling coalitions fear that any visible accommodation with Cuba could fracture their support.
```

#### BOX_004_CAND_08: economic_logistical_angle

- Actual chars: `139`
- Target chars: `200`
- Length delta: `-61`
- Rationale: Introduces economic/logistical consequences that could follow a break in relations.
- Distinctiveness: Moves the emphasis to trade/shipping disruptions rather than political agitation.

```text
causing import and shipping disruptions as governments adjust trading arrangements to avoid retaliatory measures or Cuban-linked embargoes.
```

#### BOX_004_CAND_09: uncertainty_or_source_dispute

- Actual chars: `161`
- Target chars: `200`
- Length delta: `-39`
- Rationale: Introduces uncertainty about organization/spontaneity, consistent with earlier discussion of limited effectiveness.
- Distinctiveness: Centers on contested interpretation of the nature of demonstrations.

```text
may reflect exaggerated reporting, because available accounts differ on whether demonstrations are organized or merely spontaneous reactions to the announcement.
```

#### BOX_004_CAND_10: contrarian_document_level_hypothesis

- Actual chars: `167`
- Target chars: `200`
- Length delta: `-33`
- Rationale: Contrarian document-level reading: governments signal compliance domestically without substantive alignment.
- Distinctiveness: Reframes the “majority” reaction as strategic signaling/avoidance rather than genuine mobilization.

```text
is intended primarily to demonstrate responsiveness to US policy, with governments using the Cuba issue to rally domestic support while avoiding substantive alignment.
```
