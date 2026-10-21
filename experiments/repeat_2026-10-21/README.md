# The /chat repetition penalty, ember vs mtstudio, 21 Oct 2026

Contract point 4 (LOCAL_LLM_STACK.md section 3): `repeat_penalty` p and `repeat_last_n` n. Over the
last n ids of prompt + reply so far, each distinct id outside the stop set has its logit l replaced
by l / p (l > 0) or l * p, once, before sampling.

Setup:
- **Model:** chat_sftfinal_v3 (a9d5ba5e...).
- **Prompts:** 40 frozen prompts (tinychat_hf_prompts_v1), greedy, 60 new tokens, n = 64.
- **References:** mtstudio's (2e, paperkiln 3c35e9ef working tree):
  `C:/ml_artifacts/chat_base/sftfinal/rep_p1.1_greedy.jsonl` (ab714b04...) and `rep_p1.3_greedy.jsonl`
  (b2ed0800...), both hash-checked before the run.
- **ember:** 0e5b694, built for this run. The penalty is one helper (`apply_contract_penalty`), shared by
  `generate()` and `sample_step`.

| Backend | p = 1.0 vs the recorded receipt | p = 1.1 vs mtstudio | p = 1.3 vs mtstudio |
|---|---|---|---|
| CPU | 40/40 | 40/40 | 40/40 |
| CUDA (TF32 off) | 40/40 | 40/40 | 40/40 |
| WebAssembly (`ember_chat`, 1fdaaa5 build) | | 40/40 | 40/40 |

Replies, token counts and stop reasons are all identical (`gate.log`; ember's replies in
`p*_{cpu,cuda}.jsonl`). ember-side checks before this run (338b24a):
- the rule on dumped first-token logits, 5/5;
- out-of-range requests get a 400;
- the OpenAI endpoint takes the extension fields, and `frequency_penalty`/`presence_penalty` only at 0.
