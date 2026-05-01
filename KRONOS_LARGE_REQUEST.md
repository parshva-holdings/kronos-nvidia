# Requesting Kronos-large weights

`Kronos-large` (499.2M params) is listed as **not open-source** in the model zoo. The authors sometimes share it for academic / serious research use. If you want the absolute largest base to fine-tune from, this is the email to send.

## Where to send

The paper lists corresponding-author affiliations at **Tsinghua University** and **Mohamed bin Zayed University of AI**. The cleanest channel is to open a GitHub issue — it gives you a public timestamp and the authors monitor it actively (the repo is at 22K stars and they've been responsive in 2025-26).

- **Primary**: open an issue at <https://github.com/shiyu-coder/Kronos/issues> titled "Request access to Kronos-large weights"
- **Secondary**: corresponding-author emails on the [arXiv paper](https://arxiv.org/abs/2508.02739) (page 1 footnotes)

## Email / issue template

> **Subject**: Request for access to Kronos-large weights — fine-tuning for NSE (India) research
>
> Dear Dr. Shi and co-authors,
>
> Congratulations on the AAAI 2026 acceptance — the hierarchical OHLCV tokenizer is the most elegant approach to financial time-series I've come across. I'm writing to request access to the `Kronos-large` (499M) checkpoint, which is listed as not publicly released in the model-zoo table of the README.
>
> **Use case**. I'm building an India-focused fine-tune corpus covering the full Nifty 50 index history (1996–present), all current and historical Nifty 50 / Next 50 constituents, and the Nifty 500 universe (~530 series, ~3M total daily bars). My goal is to evaluate how the largest available Kronos variant fine-tunes on a market that was likely under-represented in the global pretraining mix, and to publish a comparison report (Kronos-base vs Kronos-large, zero-shot vs fine-tuned) on the Indian benchmarks.
>
> **Compute**. I have allocated 8× H100 capacity via NVIDIA Brev for the fine-tune; this is sufficient for both stages of the upstream `train_tokenizer.py` / `train_predictor.py` pipeline within ~2 days of wall-clock time.
>
> **Privacy**. I'm happy to keep the weights strictly private (no redistribution, no derivative-model release without your prior consent) and to share the resulting evaluation report with you ahead of any publication or blog post. If you'd prefer a signed agreement, I'm fine with a one-page MOU.
>
> **About me**. <one paragraph: your role, prior published work, why this is credible — this is the part you fill in>.
>
> Could you share access — even a time-limited download link via Hugging Face's gated-repo mechanism would work — and any usage notes (preferred batch size, known finicky hyperparameters, etc.)?
>
> Many thanks for considering, and again, congratulations on a beautiful result.
>
> Best regards,
> Rohan Jain
> rohan@parshva.io

## Realistic expectations

- **If granted**: you swap `KRONOS_PRETRAIN_MODEL=NeoQuasar/Kronos-large` (or whatever the gated-repo ID is) into your `.env` and re-run `scripts/07_finetune_max.sh`. Otherwise the pipeline is identical.
- **If denied**: Kronos-base fine-tuned on the full NSE corpus is still likely to beat zero-shot Kronos-large for Nifty 50 specifically. You're not blocked.
- **Time-to-respond**: typically 1-2 weeks for academic asks. Don't wait — start the Kronos-base fine-tune in parallel.

## What about training a new larger model from scratch?

Tempting, but: pretraining a 500M-parameter Kronos-style model from scratch on 12B+ K-line records (matching the paper) takes **~$50-150K of H100-hours**, and there's no guarantee of out-performing a careful fine-tune. Skip it unless you're actually doing research.
