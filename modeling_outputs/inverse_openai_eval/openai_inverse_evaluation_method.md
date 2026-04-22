# OpenAI Inverse Candidate Evaluation

This stage consumes `modeling_outputs/inverse_openai/<document_id>/candidate_fills.jsonl`.

For each generated candidate `y_k` for box `b`, the evaluator:

1. Inserts `y_k` into box `b`.
2. Fills every other masked box with the local answer key so the forward model sees a natural completed document.
3. Runs the trained forward token classifier on the completed document.
4. Reads the mean log probability assigned to the candidate span being redacted.
5. Combines length fit, local context similarity, and forward-model likelihood.
6. Normalizes scores with a softmax over candidates for the same box.

The score is:

```text
score(y_k) =
  beta_len * log p_len(len(y_k) | observed_char_count)
+ beta_ctx * sim(y_k, visible_local_context)
+ beta_fwd * mean_i log p_theta(m_i = 1 | completed_document[y_k])
```

The normalized value is:

```text
P(y_k | document, box) = softmax_k(score(y_k) / temperature)
```

The output is not a literal Bayesian posterior yet, because the generator's
proposal distribution is not modeled. It is a calibrated finite-candidate
posterior-style distribution that is good enough for ranking, entropy, and
ground-truth similarity checks.
