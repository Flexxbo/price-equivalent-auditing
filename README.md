# Price-equivalent (PE) auditing of AI shopping agents

Code for the paper **"A Cue's Hidden Price: Auditing Seller Influence on AI Shoppers"**
by Felix Buecheler (Adelaide University), Gerardo Berbeglia (Melbourne Business School) and
Mingyu Guo (Adelaide University). Paper: arXiv link to be added.

The code runs the counterbalanced, position-controlled forced-choice audit described in
Section 3 of the paper. One product carries a seller cue (for example social proof), the
other is an otherwise identical control. The cued product's price is raised along a sweep,
and the premium at which the policy-averaged selection share falls to 0.5 is the cue's
price-equivalent (PE).

This repository contains the code, the product stimuli, the cue texts, the prompts, and the
run configurations. The per-trial logs and result files are not included yet and will be added
in a later release.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the API keys you need
```

Which keys you need depends on the experiment:

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | GPT-4o and GPT-4o mini (main audit), image generation and the vision runs |
| `AWS_BEARER_TOKEN_BEDROCK`, `BEDROCK_REGION` | Claude, Nova and Llama models via Amazon Bedrock (cross-model audit, cue rewrites) |
| `OPENROUTER_API_KEY` | Jev (decision-native model) |

The image experiment additionally needs a headless browser for rendering:
`python -m playwright install chromium`.

Every script that calls an API has a hard spend cap (`--max-budget`, plus the global caps
`LLM_AUDIT_HARD_CAP_USD` and `LLM_AUDIT_HARD_CAP_CALLS` in `.env`) and asks for confirmation
before spending unless `--yes` is given. `python src/check_cost_safety.py` verifies these
limits without any API call.

## Quick check without API cost

All runners have an offline mock mode that exercises the full pipeline:

```bash
python src/run_m4.py --mock --n 4 --yes
python src/run_robustness.py --mock --n 4 --yes
python cue_studies/run_additivity.py --mock --pairs 2
python cue_studies/run_minopt.py --mock
python cue_studies/run_wording.py --mock
python jev/run_jev_pe.py                       # estimator self-tests
python jev/run_audit.py --config jev/audit_config_v2.json --backend mock --limit-pairs 1
python src/show_prompts.py --cue sp_strong     # print the exact prompt for a condition
```

Analysis scripts ignore mock runs by design.

## Repository layout

```
src/                  measurement core (prompting, price sweep, estimators, analysis)
  experiment.py       forced choice + price sweep, counterbalancing, prompt construction
  cues.py             cue catalog (all cue sentences used in the paper)
  products.py         loads the product pairs
  models.py           model clients (OpenAI, Bedrock) and the mock model
  budget.py           cost estimation and hard spend caps
  analysis.py         PE estimators and the pair-clustered bootstrap
  nudges.py           stimulus checks (LLM-judge neutrality rubric)
cue_studies/          additivity factorial, PE-guided cue selection and the wording test
                      (uses src/ as a library)
jev/                  PE audit of the decision-native model Jev (via OpenRouter)
data/products/        the 12 product pairs (headphones, coffee machines) and the neutrality audit
```

Stimulus neutrality (App. N): the manual audit of all twelve pairs is in
`data/products/neutrality_audit.md`, the LLM-judge scores for the headphone pairs are stored in
the `qc_self_audit` field of `data/products/headphones_pairs.json`, and the judge rubric is in
`src/nudges.py`.

Results are written to `data/results/` (created on first run).

## Reproducing the paper

All commands are run from the repository root. Temperature is 0.7 throughout. The default
price grid is 0, 2, 4, 6, 8, 12, 18 and 26 percent.

| Paper | Command |
|---|---|
| Main cue battery, GPT-4o, headphones (Sec. 4.1 to 4.3) | `python src/run_m4.py --model gpt-4o --n 24 --max-budget 15` |
| Second category, coffee machines (Sec. 4.3) | `python src/run_m4.py --model gpt-4o --n 24 --category coffee --max-budget 15`, plus the domain-matched authority cue with `--conditions authority_coffee` |
| Cross-model audit (Sec. 4.4) | `python src/run_m4.py --model <alias> --n 24 --max-budget 15 --conditions sp_weak,sp_medium,sp_strong,sp_vstrong,scarcity,authority,neutral,framing_gainloss` with `<alias>` in `gpt-4o-mini, claude-sonnet, claude-haiku, nova-micro, llama-70b, llama-8b` |
| Price-only calibration (App. E) | `python src/run_m4.py --model <alias> --n 24 --conditions price_only --sweep 0,0.02,0.05,0.10,0.18,0.30,0.50,0.80` |
| Prompt-paraphrase robustness (App. J) | `python src/run_robustness.py --model gpt-4o --n 12 --max-budget 15` then `python src/analyze_robustness.py` |
| Cue-combination additivity (App. H) | `python cue_studies/run_additivity.py --eval-model gpt-4o --pairs 6 --n 24` |
| Wording of one description (Sec. 4.6) | `python cue_studies/run_wording.py --eval-model gpt-4o --gen-model claude-sonnet --cues sp_strong,scarcity --pair-index 0 --n-wordings 12 --confirm-n 30` |
| PE-guided cue selection (App. I) | `python cue_studies/run_minopt.py --eval-model gpt-4o --gen-model claude-sonnet --pairs 5 --rewrites 5 --screen-n 10 --confirm-n 24` |
| Jev audit (Sec. 4.7, App. K) | `python jev/run_audit.py --config jev/audit_config_v2.json --backend openrouter`, robustness with `jev/audit_config_v2_e1.json`, the earlier structured-cards run with `jev/audit_config_v1_cards.json`, then `--analyze-only` and `python jev/make_fig_jev.py jev_audit_v2 sp_vstrong,authority` |
| Image (presentation) cue (App. M) | `python src/images.py` (generates the product images), then `python src/experiment_multimodal.py --model gpt-4o` and `python src/probe_image_perception.py` |

Analysis of the collected runs:

```bash
python src/make_paper_numbers.py          # PEs and pair-clustered bootstrap CIs (2000 resamples, seed 0)
python src/analyze_m4.py                  # validity summary (dose-response, controls, framing)
python src/check_position_interaction.py "data/results/M4_validity/m4_<cue>_gpt-4o_*.csv"
python src/make_paper_outputs.py          # figures -> figures/, LaTeX tables -> generated/
```

`make_paper_numbers.py` produces the PE values and confidence intervals of the cue-battery,
cross-model and framing tables. It uses the model-free local-crossing estimator on the pair- and
position-averaged curve. Censored values are flagged: a curve that does not cross 0.5 within the
grid is right-censored (marked †, printed as "26+" in the paper), and a cue that is
dispreferred at equal price is floor-censored (marked ‡, printed as "≤0").

The other analysis scripts (`analyze_m4.py`, `analyze_robustness.py`, the previews written by
`make_paper_outputs.py`, and `src/analysis.py` when run on a single file) use a quick trial-level
bootstrap and serve as diagnostics. The additivity, cue-selection, wording and Jev scripts write
their own summaries to `data/results/cue_studies/` and `jev/_out/`.

Folder and file names use internal run labels: `M4` is the main audit and `E1` the
prompt-paraphrase robustness study. Identifiers containing `wtp` are historical names for the
price-equivalent, which is not a willingness to pay.

## Notes on reproducibility

The fixed seed pins stimulus construction and counterbalancing, not model sampling. Hosted
models are not deterministic, and the OpenAI models were accessed through the rolling aliases
`gpt-4o` and `gpt-4o-mini`, so a re-run measures the currently served snapshot. The exact model
identifiers used are listed in `src/models.py` and in the paper's reproducibility appendix.

The product images used in the paper's image probe are not included because of the size limit
of the supplementary material. `src/images.py` regenerates images with the same prompts
(`gpt-image-1`); generated images differ between runs.

## Citation

If you use this code, please cite:

```bibtex
@misc{buecheler2026cueprice,
  title         = {A Cue's Hidden Price: Auditing Seller Influence on AI Shoppers},
  author        = {Buecheler, Felix and Berbeglia, Gerardo and Guo, Mingyu},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## License

The code is released under the MIT License (see `LICENSE`).

## Contact

Felix Buecheler, felix.buecheler@adelaide.edu.au
