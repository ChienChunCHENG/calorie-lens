# Browser computation of the commonness prior

Inputs: `vocab_prior.json` (array aligned with vocab.json rows), `lambda` from `params.json` (0.5), `clip.json.logit_scale` (LS), embedding e (L2-normalised), text matrix T.

Per region, for every row i:
1. `cos_i = dot(T_i, e)`
2. Gate (unchanged, NO prior): `z0_i = LS*cos_i`; `p0 = softmax(z0)`; skip the region if `sum(p0[neg]) > gate.p_neg_max` or `max(p0[food]) < gate.p_food_min` (fallback regions use the stricter fallback gate as before).
3. Ranking for the mixture (food rows only): `z_i = LS*cos_i + lambda*lp_i`, where `lp_i = vocab_prior[i].lp` (centred ln prior; precomputed, so no log in the browser). Optional additional terms (e.g. image prototypes) add to `z_i` the same way.
4. `q = softmax(z_i over food rows)`; top-K / cumulative cut / "sure" threshold / mixture exactly as in snap_contract step 5 using q instead of the old renormalised food probabilities.
5. Display name: `vocab_prior[i].display ?? vocab[i].en` (zh first if present, per the existing rule).

`prior` (sum 1 over food rows) is provided for reference; `lp = ln(prior_i) - mean_food(ln prior)`; neg rows have `lp = 0` and are never ranked. Because lp is centred, lambda can be changed at runtime and lambda = 0 reproduces the current behaviour exactly.
Why the gate ignores the prior: with neutral (0) neg terms the prior changed which empty-cup crops pass (uxset negatives logging 12/80 -> 13/80, extra 1/30 -> 2/30, user 0/10); using p0 keeps the counts identical (12/80, 1/30, 0/10, drinks 39/39).
