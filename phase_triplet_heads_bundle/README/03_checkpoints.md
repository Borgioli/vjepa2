# Checkpoints + RAG indices

All checkpoint paths below are relative to the bundle root.

## V-JEPA2 backbones (~9.6 GB)

| File | Size | Notes |
|---|---|---|
| `vjepa2_1/latest.pt` | 4.8 GB | Most recent backbone snapshot (used as default by some scripts). |
| `vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt` | 4.8 GB | ViT-L distilled from ViT-G at 384 px. Referenced by the phase RAG README. |

The `latest.pt` file is **identical in size to** the surg-finetuned checkpoints; check `phase_rag/README.md` and the head config files to confirm which one a given experiment expects (the encoder settings get baked into the RAG index `metadata.json`).

## Surgically fine-tuned phase backbones (~15.3 GB)

Under `vjepa2_1/chekpoints_vjepar_surg_finetuned/` — note the original typo `chekpoints` is preserved so existing config paths still resolve.

| File | Size | Purpose |
|---|---|---|
| `single4_all_datasets_phase1_warmup_g8_latest.pth.tar` | 5.1 GB | Warmup stage. |
| `single4_all_datasets_phase2_main_g8_latest.pth.tar` | 5.1 GB | Main training stage. |
| `single4_all_datasets_phase3_higher_res_g8_latest.pth.tar` | 5.1 GB | High-res phase3 — **default for phase RAG** per `phase_rag/README.md`. |

Load with `checkpoint_key: target_encoder` at resolution 512.

## Phase-head checkpoints (~9.9 GB)

| File | Size | Notes |
|---|---|---|
| `vjepa2_1/phase3_highfps.pth.tar` | 5.1 GB | High-fps phase3 head. |
| `vjepa2_1/all_datasets_plus_surgvu_phase3_higher_res_g8_latest.pt` | 4.8 GB | All-datasets + SurgVU phase3 higher-res. |

## Phase RAG prebuilt indices (~75 MB total)

Under `vjepa2_1/phase_rag_embeddings/`.

| Index dir | Size | Built from |
|---|---|---|
| `native11_phase3_topk8_train75/` | 13 MB | phase3 high-res ckpt, top-k 8, train split |
| `native11_vitl_dist_vitg384_topk8_train75/` | 40 MB | distilled ViT-L@384 ckpt, top-k 8 |
| `native11_latestpt_topk4_train75/` | 22 MB | `latest.pt`, top-k 4 |
| `native11_phase3_topk8_train75_batch4_smoke/` | 152 KB | batch=4 smoke test |
| `native11_phase3_topk8_train75_batch8_smoke/` | 280 KB | batch=8 smoke test |
| `native11_phase3_topk8_train75_smoke/` | 56 KB | minimal smoke test |

Each index dir contains `clips.csv`, `embeddings.pt`, and a `metadata.json` that records the encoder settings — so eval/inference normally only need the index path.

## Picking the right combo

For the canonical **phase RAG eval** workflow (per `vjepa2_1/phase_rag/README.md`):

- Backbone: `chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase3_higher_res_g8_latest.pth.tar`
- Index: `phase_rag_embeddings/native11_phase3_topk8_train75/`
- CSV: `globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv`

For **OvR phase head training** (per `vjepa2_1/head_phases_ovr/README.md`):

- CSVs: `vjepa2_1/app/csv_head_models/sitl_phases_{train,val}.csv` + `globus/surgenet_phases/yt_robotic_chole_native11_phases_{train,val}.csv`
- Backbone: typically one of the surg-finetuned phase2/phase3 ckpts (check the generated config under `configs/heads_ovr/`).
