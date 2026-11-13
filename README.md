# Parameter-efficient adaptation of radiology foundation models for TB detection

Code, raw results and figures for the preprint:

> **Oni AS.** *Parameter-Efficient Adaptation of Radiology Foundation Models for Tuberculosis Detection in Low-Data Regimes: A Systematic Comparison of LoRA, BitFit, Linear Probing, and Full Fine-Tuning.* Zenodo, 2026. [doi:10.5281/zenodo.21480007](https://doi.org/10.5281/zenodo.21480007)

The study asks a practical question for low-resource hospitals: when you only have a few hundred labelled chest X-rays, which way of adapting a large pretrained model gives the most reliable TB detector, and which one still works on X-rays from other hospitals?

## What was done

- **Backbones:** [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (ViT-B/16) and [RAD-DINO](https://huggingface.co/microsoft/rad-dino) (ViT-B/14).
- **Adaptation methods:** linear probing, BitFit (bias terms only), LoRA at ranks 4, 8 and 16 (on Q, K, V and the output projection for both models), and full fine-tuning.
- **Training data:** Shenzhen (662 images; 336 TB, 326 normal), 5-fold stratified cross-validation (529 or 530 training images per fold).
- **External test sets, never seen in training:** Montgomery (USA, 138 images), TBX11K (China, 8,811 images; TB vs healthy only) and the Kaggle TB Chest Radiography Database (multi-source, 4,200 images).
- **Preprocessing ablation:** with and without CLAHE contrast enhancement for the best configuration.
- **Equal budget for every configuration:** up to 50 epochs, early stopping (patience 15) on validation AUROC, focal loss with class weights, AdamW and cosine schedule, mixed precision, seed 42.
- **Runs:** 13 configurations × 5 folds = 65 training runs on one NVIDIA A40 (RunPod), plus 195 external evaluations (65 fold models × 3 external sets).

## Main results

Internal results are the mean over the 5 Shenzhen validation folds. External results are the mean AUROC of the 5 fold models on each external set.

| Configuration | Trainable params | Internal AUROC | Montgomery | TBX11K | Kaggle TB |
|---|---:|---:|---:|---:|---:|
| RAD-DINO + LoRA r=16, no CLAHE | 1.18 M (1.35%) | **0.987** ± 0.003 | **0.961** | **0.935** | **0.929** |
| RAD-DINO + LoRA r=16 | 1.18 M (1.35%) | 0.973 ± 0.012 | 0.930 | 0.930 | 0.901 |
| RAD-DINO + LoRA r=8 | 0.59 M (0.68%) | 0.974 ± 0.009 | 0.924 | 0.933 | 0.892 |
| RAD-DINO + BitFit | 0.11 M (0.12%) | 0.963 ± 0.012 | 0.917 | 0.932 | 0.880 |
| RAD-DINO + full fine-tuning | 86.6 M (100%) | 0.974 ± 0.009 | 0.880 | 0.916 | 0.862 |
| RAD-DINO + linear probe | 3 K | 0.949 ± 0.015 | 0.882 | 0.905 | 0.867 |
| BiomedCLIP + full fine-tuning | 86.2 M (100%) | 0.966 ± 0.018 | 0.903 | 0.810 | 0.903 |
| BiomedCLIP + LoRA r=16 | 0.93 M (1.07%) | 0.957 ± 0.025 | 0.903 | 0.802 | 0.903 |

All rows except the first use CLAHE. The full table for all 13 configurations is printed by `scripts/reproduce_tables.py`.

Key findings:

1. **LoRA transferred better than full fine-tuning.** With identical preprocessing (CLAHE), RAD-DINO + LoRA r=16 kept 0.930 AUROC on Montgomery while full fine-tuning fell to 0.880, while training 1.35% of the parameters.
2. **RAD-DINO generalised better than BiomedCLIP,** most clearly on TBX11K (0.90–0.94 vs 0.75–0.81).
3. **Preprocessing mattered.** Removing CLAHE from the best configuration improved both internal and every external result.
4. **Clinically useful operating points were rarely reached.** Only one configuration/dataset pair (RAD-DINO + LoRA r=16 without CLAHE, on Montgomery: 0.862) reached ≥ 0.85 sensitivity at 95% specificity on external data.

## Check the numbers yourself (no GPU needed)

Every table above is recomputed from the raw per-run logs:

```bash
pip install pandas numpy
python scripts/reproduce_tables.py
```

The script rebuilds the internal and external summaries from `results/raw/`, writes them to `results/reproduced/`, and checks them against the tables saved at the time of the runs.

## Repository layout

```
notebooks/tb_peft_runpod.ipynb   The notebook that produced every result, with its original outputs
results/raw/                     Per-run logs: 65 training runs (with epoch-by-epoch history) and 195 external evaluations
results/tables/                  Summary tables saved at the time of the runs
figures/                         Figures from the preprint
scripts/reproduce_tables.py      Rebuilds and checks the summary tables from the raw logs
```

## Re-running the experiments

The notebook was run top to bottom on a RunPod A40 (48 GB) with PyTorch 2.4.1 / CUDA 12.4 and Python 3.11. It expects the datasets under `/workspace` (paths are set in the hyperparameters cell).

1. Install the dependencies: `pip install -r requirements.txt`
2. Set your Kaggle API credentials as environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`) so the notebook can download the public datasets.
3. The notebook downloads all datasets from Kaggle: Shenzhen and Montgomery (`kmader/pulmonary-chest-xray-abnormalities`, originally released by the [U.S. National Library of Medicine](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)), TBX11K (`vbookshelf/tbx11k-simplified`) and the TB Chest Radiography Database (`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`). Check the dataset paths in the hyperparameters cell after download.
4. Run the notebook. The full grid takes roughly 30 GPU-hours.

Model weights and images are not included in this repository; the images are subject to their original licences.

## Limitations

- **Labels:** all four datasets use their published image-level labels; none were re-read by a radiologist for this study.
- **Small training set:** a single training site (Shenzhen) with about 530 images per fold.
- **Internal estimates are optimistic:** the internal AUROC is the best validation AUROC across epochs on the same fold used for early stopping. External results do not have this issue, because the saved fold models were evaluated once on data they never saw.
- **Internal PR-AUC and sensitivity are computed with "normal" as the positive class.** In the training loop the positive class after label remapping is "normal", not TB. AUROC is unaffected, but the internal PR-AUC and "sensitivity at fixed specificity" columns in `results/raw/experiment_results.json` describe normal-image detection. All external metrics treat TB as the positive class and are correct. This will be corrected in a revised version of the preprint.
- **The CLAHE ablation was run for one configuration only** (RAD-DINO + LoRA r=16), so the best overall result comes from the one configuration tested without CLAHE.
- **Single seed** (42) for all runs; variation is reported across folds only.

## Citation

```bibtex
@misc{oni2026tbpeft,
  author    = {Oni, Ayotunde Samuel},
  title     = {Parameter-Efficient Adaptation of Radiology Foundation Models for Tuberculosis Detection in Low-Data Regimes: A Systematic Comparison of LoRA, BitFit, Linear Probing, and Full Fine-Tuning},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21480007}
}
```

## Author

Dr Ayotunde Samuel Oni, MBBS (Ibadan) — medical doctor and machine learning engineer, Ibadan, Nigeria.
[LinkedIn](https://www.linkedin.com/in/ayotundeoni) · [GitHub](https://github.com/AyotundeOni)

## Licence

Code and result files are released under the MIT licence (see `LICENSE`).
