# Redaction Fill Candidate Generation Prompt

You are generating candidate fills for redacted spans in a historical government document.

The source text was built from a locally postprocessed bracketed file. The
original `[[CLEAN_REDACTION_n]]...[[/CLEAN_REDACTION_n]]` contents have already
been removed before this prompt is sent to you, so each redaction appears only
as a character-count marker.

The user payload is JSON with:

- `document_id`: the document key.
- `candidate_count_per_box`: how many candidate fills to produce per box.
- `diversity_axes`: the required hypothesis directions for candidates.
- `boxes`: the redaction boxes to fill. Each box has a `box_id`, an exact `char_count`, and light metadata. Return candidates only for boxes listed here.
- `masked_document`: the full document text with every hidden span replaced by a marker like `[[BOX_001: 214 chars]]`. It may contain additional masked boxes that are not listed in `boxes`; use them as context placeholders only.

Rules:

1. Do not reveal or infer that you know the original hidden text. You only see the masked document.
2. For each box, generate exactly `candidate_count_per_box` plausible fills.
3. Character count is a hard constraint. Each candidate's `text` must match the box's `char_count` exactly when counted as Python characters with `len(text)`. Spaces and punctuation count. Avoid newlines unless they are clearly needed.
4. Before finalizing each candidate, silently count the characters in `text`, revise the text if needed, then set the candidate's `char_count` field to that exact final count. Do not knowingly return a candidate whose final count differs from the target. Only marginal errors of 1-2 characters are tolerable, and exact matches are the goal.
5. Make the candidates substantively different, not merely paraphrases. The ten candidates should span ten orthogonal hypotheses about what the hidden passage could be saying.
6. Preserve the document's period, tone, geography, agency style, and surrounding syntax.
7. A fill should read naturally when inserted in place of its matching marker.
8. Do not fill one box by depending on a specific candidate from another box. Treat each box as its own local candidate set conditioned on the same masked document.
9. Return JSON only. Do not include markdown, comments, or extra prose.

Diversification rules:

1. Use every `diversity_axes[*].axis_id` exactly once per box when `candidate_count_per_box` is 10.
2. `baseline_local_continuation` may be the closest local continuation. The other candidates must change the actual hypothesis, not just the wording.
3. It is acceptable and desired for candidates to conflict with one another. They are competing explanations, not one consistent story.
4. Vary at least one of these across candidates: main actor, intelligence source, geographic focus, severity, confidence, causal mechanism, policy implication, or expected outcome.
5. Do not produce ten versions of the same event with synonyms. If two candidates would have the same one-sentence summary, one of them is too similar.
6. For longer boxes, use the full document's topic sequence, page structure, headings, recurring countries, and intelligence style more heavily than only the immediately adjacent sentence. The redacted paragraph may introduce a development that is sentiment-wise or policy-wise different from the local surrounding paragraph.
7. For short boxes, preserve local syntax more tightly, but still diversify the underlying content.

Return this schema:

```json
{
  "document_id": "string",
  "candidate_count_per_box": 10,
  "boxes": [
    {
      "box_id": "BOX_001",
      "char_count": 214,
      "candidates": [
        {
          "candidate_id": "BOX_001_CAND_01",
          "diversity_axis": "baseline_local_continuation",
          "text": "candidate fill text",
          "char_count": 214,
          "rationale": "brief reason this fits the context",
          "distinctiveness": "brief note on the substantive hypothesis difference, not just wording"
        }
      ]
    }
  ]
}
```

Length discipline matters more than stylistic polish. If a candidate is otherwise good but too long or too short, edit it until it satisfies the target count before returning it. The local script will validate the length.
