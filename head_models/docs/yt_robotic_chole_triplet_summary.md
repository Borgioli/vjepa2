# yt_robotic_chole_triplets

## Assumptions
- Output root: `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets`
- Exported annotation family: `triplet labels`
- Taxonomy option used: `A (native labels)`
- Clip construction rule: fixed 16-frame clips cut independently from contiguous runs of each triplet label; overlapping triplets are exported separately.
- Maximum clip duration: 4.00 seconds at 4.0 fps.
- Clip window rule: left-aligned non-overlapping 16-frame windows within each triplet's active runs; leftover tails shorter than 16 frames are dropped.
- Multiple triplets may be active at the same time; overlapping mapped labels are stored in the multi-label split CSVs.
- Split strategy: video-disjoint exhaustive search over candidate validation video sets.
- CSV sync rule: source split paths are normalized to this local directory; existing multi-label rows are preserved, and missing multi-label rows are added from the one-label split CSVs plus mapped annotation overlaps when available.

## Selected Val Videos
- yt_robotic_chole_Batch0
- yt_robotic_chole_Batch2
- yt_robotic_chole_Batch6

## CSV Coverage
- Local clip files: 5371
- Files covered by one-label split CSVs: 5371 / 5371
- Files covered by multi-label split CSVs: 5371 / 5371

| split | one-label rows | one-label unique clips | multi-label rows | multi-label unique clips | multi rows added | missing one-label files | missing multi-label files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3567 | 3567 | 3567 | 3567 | 155 | 0 | 0 |
| val | 1804 | 1804 | 1804 | 1804 | 94 | 0 | 0 |

## Final Label Mapping
- 0: bipolar | coagulation | connective tissue
- 1: bipolar | coagulation | cystic duct
- 2: bipolar | grasp/retract | adhesion
- 3: bipolar | grasp/retract | connective tissue
- 4: bipolar | grasp/retract | cystic duct
- 5: bipolar | grasp/retract | cystic pedicle
- 6: bipolar | grasp/retract | gallbladder
- 7: bipolar | grasp/retract | gallbladder wall
- 8: bipolar | grasp/retract | liver
- 9: bipolar | grasp/retract | null
- 10: bipolar | grasp/retract | peritoneum
- 11: bipolar | grasp/retract | suture
- 12: bipolar | null | cystic pedicle
- 13: bipolar | null | null
- 14: bipolar | null | peritoneum
- 15: clipper | clip | cystic artery
- 16: clipper | clip | cystic duct
- 17: clipper | null | cystic duct
- 18: clipper | null | liver
- 19: grasper | grasp/retract | adhesion
- 20: grasper | grasp/retract | connective tissue
- 21: grasper | grasp/retract | cystic artery
- 22: grasper | grasp/retract | cystic duct
- 23: grasper | grasp/retract | cystic pedicle
- 24: grasper | grasp/retract | fallciform ligament
- 25: grasper | grasp/retract | gallbladder
- 26: grasper | grasp/retract | gallbladder wall
- 27: grasper | grasp/retract | gut
- 28: grasper | grasp/retract | liver
- 29: grasper | grasp/retract | null
- 30: grasper | grasp/retract | omentum
- 31: grasper | grasp/retract | peritoneum
- 32: grasper | grasp/retract | specimen bag
- 33: grasper | grasp/retract | suture
- 34: grasper | null | connective tissue
- 35: grasper | null | cystic artery
- 36: grasper | null | null
- 37: harmonic shears | cut | adhesion
- 38: harmonic shears | cut | connective tissue
- 39: harmonic shears | cut | cystic duct
- 40: harmonic shears | cut | cystic pedicle
- 41: harmonic shears | cut | peritoneum
- 42: harmonic shears | dissect | cystic duct
- 43: harmonic shears | dissect | cystic pedicle
- 44: harmonic shears | null | adhesion
- 45: harmonic shears | null | gallbladder
- 46: harmonic shears | null | liver
- 47: harmonic shears | null | suture
- 48: hook | coagulation | adhesion
- 49: hook | coagulation | connective tissue
- 50: hook | coagulation | cystic artery
- 51: hook | coagulation | cystic pedicle
- 52: hook | coagulation | gallbladder wall
- 53: hook | coagulation | peritoneum
- 54: hook | cut | cystic artery
- 55: hook | cut | cystic duct
- 56: hook | cut | gallbladder wall
- 57: hook | dissect | cystic artery
- 58: hook | dissect | cystic duct
- 59: hook | dissect | cystic pedicle
- 60: hook | null | connective tissue
- 61: hook | null | cystic duct
- 62: hook | null | gallbladder
- 63: hook | null | gallbladder wall
- 64: hook | null | gut
- 65: hook | null | liver
- 66: hook | null | null
- 67: irrigator | clean | fluid
- 68: irrigator | grasp/retract | adhesion
- 69: irrigator | grasp/retract | connective tissue
- 70: irrigator | grasp/retract | cystic artery
- 71: irrigator | grasp/retract | cystic duct
- 72: irrigator | grasp/retract | cystic pedicle
- 73: irrigator | grasp/retract | gallbladder
- 74: irrigator | grasp/retract | liver
- 75: irrigator | grasp/retract | peritoneum
- 76: irrigator | null | adhesion
- 77: irrigator | null | fluid
- 78: irrigator | null | gallbladder
- 79: irrigator | null | liver
- 80: needle driver | null | cystic duct
- 81: needle driver | null | suture
- 82: scissors | coagulation | adhesion
- 83: scissors | coagulation | connective tissue
- 84: scissors | coagulation | cystic artery
- 85: scissors | coagulation | cystic duct
- 86: scissors | coagulation | gallbladder wall
- 87: scissors | coagulation | peritoneum
- 88: scissors | cut | adhesion
- 89: scissors | cut | cystic artery
- 90: scissors | cut | cystic duct
- 91: scissors | cut | suture
- 92: scissors | dissect | adhesion
- 93: scissors | dissect | cystic artery
- 94: scissors | dissect | cystic duct
- 95: scissors | dissect | cystic pedicle
- 96: scissors | null | adhesion
- 97: scissors | null | connective tissue
- 98: scissors | null | gallbladder
- 99: scissors | null | gallbladder wall
- 100: scissors | null | null
- 101: scissors | null | peritoneum
- 102: stapler | null | cystic duct

## Train/Val Balance
| id | label | train_count | train_pct | val_count | val_pct |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | bipolar \| coagulation \| connective tissue | 7 | 0.20% | 1 | 0.06% |
| 1 | bipolar \| coagulation \| cystic duct | 3 | 0.08% | 0 | 0.00% |
| 2 | bipolar \| grasp/retract \| adhesion | 8 | 0.22% | 8 | 0.44% |
| 3 | bipolar \| grasp/retract \| connective tissue | 8 | 0.22% | 1 | 0.06% |
| 4 | bipolar \| grasp/retract \| cystic duct | 10 | 0.28% | 6 | 0.33% |
| 5 | bipolar \| grasp/retract \| cystic pedicle | 6 | 0.17% | 6 | 0.33% |
| 6 | bipolar \| grasp/retract \| gallbladder | 94 | 2.64% | 41 | 2.27% |
| 7 | bipolar \| grasp/retract \| gallbladder wall | 5 | 0.14% | 2 | 0.11% |
| 8 | bipolar \| grasp/retract \| liver | 1 | 0.03% | 0 | 0.00% |
| 9 | bipolar \| grasp/retract \| null | 1 | 0.03% | 0 | 0.00% |
| 10 | bipolar \| grasp/retract \| peritoneum | 0 | 0.00% | 3 | 0.17% |
| 11 | bipolar \| grasp/retract \| suture | 12 | 0.34% | 3 | 0.17% |
| 12 | bipolar \| null \| cystic pedicle | 19 | 0.53% | 0 | 0.00% |
| 13 | bipolar \| null \| null | 1 | 0.03% | 0 | 0.00% |
| 14 | bipolar \| null \| peritoneum | 1 | 0.03% | 0 | 0.00% |
| 15 | clipper \| clip \| cystic artery | 5 | 0.14% | 2 | 0.11% |
| 16 | clipper \| clip \| cystic duct | 17 | 0.48% | 8 | 0.44% |
| 17 | clipper \| null \| cystic duct | 1 | 0.03% | 0 | 0.00% |
| 18 | clipper \| null \| liver | 20 | 0.56% | 3 | 0.17% |
| 19 | grasper \| grasp/retract \| adhesion | 55 | 1.54% | 57 | 3.16% |
| 20 | grasper \| grasp/retract \| connective tissue | 25 | 0.70% | 3 | 0.17% |
| 21 | grasper \| grasp/retract \| cystic artery | 3 | 0.08% | 129 | 7.15% |
| 22 | grasper \| grasp/retract \| cystic duct | 77 | 2.16% | 6 | 0.33% |
| 23 | grasper \| grasp/retract \| cystic pedicle | 12 | 0.34% | 22 | 1.22% |
| 24 | grasper \| grasp/retract \| fallciform ligament | 2 | 0.06% | 0 | 0.00% |
| 25 | grasper \| grasp/retract \| gallbladder | 1100 | 30.84% | 498 | 27.61% |
| 26 | grasper \| grasp/retract \| gallbladder wall | 8 | 0.22% | 4 | 0.22% |
| 27 | grasper \| grasp/retract \| gut | 1 | 0.03% | 0 | 0.00% |
| 28 | grasper \| grasp/retract \| liver | 200 | 5.61% | 119 | 6.60% |
| 29 | grasper \| grasp/retract \| null | 51 | 1.43% | 24 | 1.33% |
| 30 | grasper \| grasp/retract \| omentum | 11 | 0.31% | 29 | 1.61% |
| 31 | grasper \| grasp/retract \| peritoneum | 3 | 0.08% | 2 | 0.11% |
| 32 | grasper \| grasp/retract \| specimen bag | 1 | 0.03% | 1 | 0.06% |
| 33 | grasper \| grasp/retract \| suture | 99 | 2.78% | 50 | 2.77% |
| 34 | grasper \| null \| connective tissue | 1 | 0.03% | 0 | 0.00% |
| 35 | grasper \| null \| cystic artery | 1 | 0.03% | 0 | 0.00% |
| 36 | grasper \| null \| null | 42 | 1.18% | 11 | 0.61% |
| 37 | harmonic shears \| cut \| adhesion | 7 | 0.20% | 3 | 0.17% |
| 38 | harmonic shears \| cut \| connective tissue | 48 | 1.35% | 0 | 0.00% |
| 39 | harmonic shears \| cut \| cystic duct | 1 | 0.03% | 2 | 0.11% |
| 40 | harmonic shears \| cut \| cystic pedicle | 1 | 0.03% | 2 | 0.11% |
| 41 | harmonic shears \| cut \| peritoneum | 10 | 0.28% | 0 | 0.00% |
| 42 | harmonic shears \| dissect \| cystic duct | 6 | 0.17% | 0 | 0.00% |
| 43 | harmonic shears \| dissect \| cystic pedicle | 5 | 0.14% | 0 | 0.00% |
| 44 | harmonic shears \| null \| adhesion | 1 | 0.03% | 0 | 0.00% |
| 45 | harmonic shears \| null \| gallbladder | 4 | 0.11% | 0 | 0.00% |
| 46 | harmonic shears \| null \| liver | 1 | 0.03% | 0 | 0.00% |
| 47 | harmonic shears \| null \| suture | 1 | 0.03% | 0 | 0.00% |
| 48 | hook \| coagulation \| adhesion | 87 | 2.44% | 86 | 4.77% |
| 49 | hook \| coagulation \| connective tissue | 453 | 12.70% | 201 | 11.14% |
| 50 | hook \| coagulation \| cystic artery | 0 | 0.00% | 3 | 0.17% |
| 51 | hook \| coagulation \| cystic pedicle | 4 | 0.11% | 6 | 0.33% |
| 52 | hook \| coagulation \| gallbladder wall | 8 | 0.22% | 4 | 0.22% |
| 53 | hook \| coagulation \| peritoneum | 65 | 1.82% | 18 | 1.00% |
| 54 | hook \| cut \| cystic artery | 2 | 0.06% | 0 | 0.00% |
| 55 | hook \| cut \| cystic duct | 3 | 0.08% | 1 | 0.06% |
| 56 | hook \| cut \| gallbladder wall | 0 | 0.00% | 1 | 0.06% |
| 57 | hook \| dissect \| cystic artery | 33 | 0.93% | 17 | 0.94% |
| 58 | hook \| dissect \| cystic duct | 73 | 2.05% | 28 | 1.55% |
| 59 | hook \| dissect \| cystic pedicle | 169 | 4.74% | 67 | 3.71% |
| 60 | hook \| null \| connective tissue | 2 | 0.06% | 0 | 0.00% |
| 61 | hook \| null \| cystic duct | 3 | 0.08% | 0 | 0.00% |
| 62 | hook \| null \| gallbladder | 3 | 0.08% | 7 | 0.39% |
| 63 | hook \| null \| gallbladder wall | 3 | 0.08% | 0 | 0.00% |
| 64 | hook \| null \| gut | 3 | 0.08% | 0 | 0.00% |
| 65 | hook \| null \| liver | 1 | 0.03% | 2 | 0.11% |
| 66 | hook \| null \| null | 2 | 0.06% | 2 | 0.11% |
| 67 | irrigator \| clean \| fluid | 179 | 5.02% | 139 | 7.71% |
| 68 | irrigator \| grasp/retract \| adhesion | 2 | 0.06% | 3 | 0.17% |
| 69 | irrigator \| grasp/retract \| connective tissue | 2 | 0.06% | 0 | 0.00% |
| 70 | irrigator \| grasp/retract \| cystic artery | 2 | 0.06% | 0 | 0.00% |
| 71 | irrigator \| grasp/retract \| cystic duct | 2 | 0.06% | 0 | 0.00% |
| 72 | irrigator \| grasp/retract \| cystic pedicle | 8 | 0.22% | 0 | 0.00% |
| 73 | irrigator \| grasp/retract \| gallbladder | 38 | 1.07% | 2 | 0.11% |
| 74 | irrigator \| grasp/retract \| liver | 12 | 0.34% | 8 | 0.44% |
| 75 | irrigator \| grasp/retract \| peritoneum | 2 | 0.06% | 0 | 0.00% |
| 76 | irrigator \| null \| adhesion | 4 | 0.11% | 0 | 0.00% |
| 77 | irrigator \| null \| fluid | 0 | 0.00% | 1 | 0.06% |
| 78 | irrigator \| null \| gallbladder | 1 | 0.03% | 0 | 0.00% |
| 79 | irrigator \| null \| liver | 18 | 0.50% | 0 | 0.00% |
| 80 | needle driver \| null \| cystic duct | 0 | 0.00% | 2 | 0.11% |
| 81 | needle driver \| null \| suture | 56 | 1.57% | 7 | 0.39% |
| 82 | scissors \| coagulation \| adhesion | 53 | 1.49% | 58 | 3.22% |
| 83 | scissors \| coagulation \| connective tissue | 99 | 2.78% | 64 | 3.55% |
| 84 | scissors \| coagulation \| cystic artery | 1 | 0.03% | 0 | 0.00% |
| 85 | scissors \| coagulation \| cystic duct | 1 | 0.03% | 0 | 0.00% |
| 86 | scissors \| coagulation \| gallbladder wall | 2 | 0.06% | 2 | 0.11% |
| 87 | scissors \| coagulation \| peritoneum | 15 | 0.42% | 1 | 0.06% |
| 88 | scissors \| cut \| adhesion | 4 | 0.11% | 4 | 0.22% |
| 89 | scissors \| cut \| cystic artery | 0 | 0.00% | 1 | 0.06% |
| 90 | scissors \| cut \| cystic duct | 7 | 0.20% | 1 | 0.06% |
| 91 | scissors \| cut \| suture | 0 | 0.00% | 2 | 0.11% |
| 92 | scissors \| dissect \| adhesion | 0 | 0.00% | 1 | 0.06% |
| 93 | scissors \| dissect \| cystic artery | 5 | 0.14% | 4 | 0.22% |
| 94 | scissors \| dissect \| cystic duct | 14 | 0.39% | 6 | 0.33% |
| 95 | scissors \| dissect \| cystic pedicle | 47 | 1.32% | 7 | 0.39% |
| 96 | scissors \| null \| adhesion | 23 | 0.64% | 0 | 0.00% |
| 97 | scissors \| null \| connective tissue | 45 | 1.26% | 0 | 0.00% |
| 98 | scissors \| null \| gallbladder | 2 | 0.06% | 1 | 0.06% |
| 99 | scissors \| null \| gallbladder wall | 1 | 0.03% | 1 | 0.06% |
| 100 | scissors \| null \| null | 2 | 0.06% | 0 | 0.00% |
| 101 | scissors \| null \| peritoneum | 9 | 0.25% | 0 | 0.00% |
| 102 | stapler \| null \| cystic duct | 1 | 0.03% | 0 | 0.00% |

## Drop Summary
- Kept fixed-length clips present in local directory: 5371
- Dropped unlabeled runs: not recomputed during this CSV/path sync.
- Dropped ambiguous overlap runs: not recomputed during this CSV/path sync.
- Dropped short labeled runs (<16 frames): not recomputed during this CSV/path sync.
- Dropped leftover tail fragments after chunking: not recomputed during this CSV/path sync.
- Dropped unmapped split rows: 0

## Imbalance Warnings
- Train rare classes below 5%: bipolar | coagulation | connective tissue (7/3567, 0.20%), bipolar | coagulation | cystic duct (3/3567, 0.08%), bipolar | grasp/retract | adhesion (8/3567, 0.22%), bipolar | grasp/retract | connective tissue (8/3567, 0.22%), bipolar | grasp/retract | cystic duct (10/3567, 0.28%), bipolar | grasp/retract | cystic pedicle (6/3567, 0.17%), bipolar | grasp/retract | gallbladder (94/3567, 2.64%), bipolar | grasp/retract | gallbladder wall (5/3567, 0.14%), bipolar | grasp/retract | liver (1/3567, 0.03%), bipolar | grasp/retract | null (1/3567, 0.03%), bipolar | grasp/retract | suture (12/3567, 0.34%), bipolar | null | cystic pedicle (19/3567, 0.53%), bipolar | null | null (1/3567, 0.03%), bipolar | null | peritoneum (1/3567, 0.03%), clipper | clip | cystic artery (5/3567, 0.14%), clipper | clip | cystic duct (17/3567, 0.48%), clipper | null | cystic duct (1/3567, 0.03%), clipper | null | liver (20/3567, 0.56%), grasper | grasp/retract | adhesion (55/3567, 1.54%), grasper | grasp/retract | connective tissue (25/3567, 0.70%), grasper | grasp/retract | cystic artery (3/3567, 0.08%), grasper | grasp/retract | cystic duct (77/3567, 2.16%), grasper | grasp/retract | cystic pedicle (12/3567, 0.34%), grasper | grasp/retract | fallciform ligament (2/3567, 0.06%), grasper | grasp/retract | gallbladder wall (8/3567, 0.22%), grasper | grasp/retract | gut (1/3567, 0.03%), grasper | grasp/retract | null (51/3567, 1.43%), grasper | grasp/retract | omentum (11/3567, 0.31%), grasper | grasp/retract | peritoneum (3/3567, 0.08%), grasper | grasp/retract | specimen bag (1/3567, 0.03%), grasper | grasp/retract | suture (99/3567, 2.78%), grasper | null | connective tissue (1/3567, 0.03%), grasper | null | cystic artery (1/3567, 0.03%), grasper | null | null (42/3567, 1.18%), harmonic shears | cut | adhesion (7/3567, 0.20%), harmonic shears | cut | connective tissue (48/3567, 1.35%), harmonic shears | cut | cystic duct (1/3567, 0.03%), harmonic shears | cut | cystic pedicle (1/3567, 0.03%), harmonic shears | cut | peritoneum (10/3567, 0.28%), harmonic shears | dissect | cystic duct (6/3567, 0.17%), harmonic shears | dissect | cystic pedicle (5/3567, 0.14%), harmonic shears | null | adhesion (1/3567, 0.03%), harmonic shears | null | gallbladder (4/3567, 0.11%), harmonic shears | null | liver (1/3567, 0.03%), harmonic shears | null | suture (1/3567, 0.03%), hook | coagulation | adhesion (87/3567, 2.44%), hook | coagulation | cystic pedicle (4/3567, 0.11%), hook | coagulation | gallbladder wall (8/3567, 0.22%), hook | coagulation | peritoneum (65/3567, 1.82%), hook | cut | cystic artery (2/3567, 0.06%), hook | cut | cystic duct (3/3567, 0.08%), hook | dissect | cystic artery (33/3567, 0.93%), hook | dissect | cystic duct (73/3567, 2.05%), hook | dissect | cystic pedicle (169/3567, 4.74%), hook | null | connective tissue (2/3567, 0.06%), hook | null | cystic duct (3/3567, 0.08%), hook | null | gallbladder (3/3567, 0.08%), hook | null | gallbladder wall (3/3567, 0.08%), hook | null | gut (3/3567, 0.08%), hook | null | liver (1/3567, 0.03%), hook | null | null (2/3567, 0.06%), irrigator | grasp/retract | adhesion (2/3567, 0.06%), irrigator | grasp/retract | connective tissue (2/3567, 0.06%), irrigator | grasp/retract | cystic artery (2/3567, 0.06%), irrigator | grasp/retract | cystic duct (2/3567, 0.06%), irrigator | grasp/retract | cystic pedicle (8/3567, 0.22%), irrigator | grasp/retract | gallbladder (38/3567, 1.07%), irrigator | grasp/retract | liver (12/3567, 0.34%), irrigator | grasp/retract | peritoneum (2/3567, 0.06%), irrigator | null | adhesion (4/3567, 0.11%), irrigator | null | gallbladder (1/3567, 0.03%), irrigator | null | liver (18/3567, 0.50%), needle driver | null | suture (56/3567, 1.57%), scissors | coagulation | adhesion (53/3567, 1.49%), scissors | coagulation | connective tissue (99/3567, 2.78%), scissors | coagulation | cystic artery (1/3567, 0.03%), scissors | coagulation | cystic duct (1/3567, 0.03%), scissors | coagulation | gallbladder wall (2/3567, 0.06%), scissors | coagulation | peritoneum (15/3567, 0.42%), scissors | cut | adhesion (4/3567, 0.11%), scissors | cut | cystic duct (7/3567, 0.20%), scissors | dissect | cystic artery (5/3567, 0.14%), scissors | dissect | cystic duct (14/3567, 0.39%), scissors | dissect | cystic pedicle (47/3567, 1.32%), scissors | null | adhesion (23/3567, 0.64%), scissors | null | connective tissue (45/3567, 1.26%), scissors | null | gallbladder (2/3567, 0.06%), scissors | null | gallbladder wall (1/3567, 0.03%), scissors | null | null (2/3567, 0.06%), scissors | null | peritoneum (9/3567, 0.25%), stapler | null | cystic duct (1/3567, 0.03%)
- Val rare classes below 5%: bipolar | coagulation | connective tissue (1/1804, 0.06%), bipolar | grasp/retract | adhesion (8/1804, 0.44%), bipolar | grasp/retract | connective tissue (1/1804, 0.06%), bipolar | grasp/retract | cystic duct (6/1804, 0.33%), bipolar | grasp/retract | cystic pedicle (6/1804, 0.33%), bipolar | grasp/retract | gallbladder (41/1804, 2.27%), bipolar | grasp/retract | gallbladder wall (2/1804, 0.11%), bipolar | grasp/retract | peritoneum (3/1804, 0.17%), bipolar | grasp/retract | suture (3/1804, 0.17%), clipper | clip | cystic artery (2/1804, 0.11%), clipper | clip | cystic duct (8/1804, 0.44%), clipper | null | liver (3/1804, 0.17%), grasper | grasp/retract | adhesion (57/1804, 3.16%), grasper | grasp/retract | connective tissue (3/1804, 0.17%), grasper | grasp/retract | cystic duct (6/1804, 0.33%), grasper | grasp/retract | cystic pedicle (22/1804, 1.22%), grasper | grasp/retract | gallbladder wall (4/1804, 0.22%), grasper | grasp/retract | null (24/1804, 1.33%), grasper | grasp/retract | omentum (29/1804, 1.61%), grasper | grasp/retract | peritoneum (2/1804, 0.11%), grasper | grasp/retract | specimen bag (1/1804, 0.06%), grasper | grasp/retract | suture (50/1804, 2.77%), grasper | null | null (11/1804, 0.61%), harmonic shears | cut | adhesion (3/1804, 0.17%), harmonic shears | cut | cystic duct (2/1804, 0.11%), harmonic shears | cut | cystic pedicle (2/1804, 0.11%), hook | coagulation | adhesion (86/1804, 4.77%), hook | coagulation | cystic artery (3/1804, 0.17%), hook | coagulation | cystic pedicle (6/1804, 0.33%), hook | coagulation | gallbladder wall (4/1804, 0.22%), hook | coagulation | peritoneum (18/1804, 1.00%), hook | cut | cystic duct (1/1804, 0.06%), hook | cut | gallbladder wall (1/1804, 0.06%), hook | dissect | cystic artery (17/1804, 0.94%), hook | dissect | cystic duct (28/1804, 1.55%), hook | dissect | cystic pedicle (67/1804, 3.71%), hook | null | gallbladder (7/1804, 0.39%), hook | null | liver (2/1804, 0.11%), hook | null | null (2/1804, 0.11%), irrigator | grasp/retract | adhesion (3/1804, 0.17%), irrigator | grasp/retract | gallbladder (2/1804, 0.11%), irrigator | grasp/retract | liver (8/1804, 0.44%), irrigator | null | fluid (1/1804, 0.06%), needle driver | null | cystic duct (2/1804, 0.11%), needle driver | null | suture (7/1804, 0.39%), scissors | coagulation | adhesion (58/1804, 3.22%), scissors | coagulation | connective tissue (64/1804, 3.55%), scissors | coagulation | gallbladder wall (2/1804, 0.11%), scissors | coagulation | peritoneum (1/1804, 0.06%), scissors | cut | adhesion (4/1804, 0.22%), scissors | cut | cystic artery (1/1804, 0.06%), scissors | cut | cystic duct (1/1804, 0.06%), scissors | cut | suture (2/1804, 0.11%), scissors | dissect | adhesion (1/1804, 0.06%), scissors | dissect | cystic artery (4/1804, 0.22%), scissors | dissect | cystic duct (6/1804, 0.33%), scissors | dissect | cystic pedicle (7/1804, 0.39%), scissors | null | gallbladder (1/1804, 0.06%), scissors | null | gallbladder wall (1/1804, 0.06%)

## Validation Checks
- Train samples: 3567
- Val samples: 1804
- Multi-label train rows: 3567
- Multi-label val rows: 1804
- Every exported clip path in the split CSVs points to an existing `.mp4` file under the local output root.
- All split paths are absolute and space-free.
- Split files use a single space delimiter and contain no header row.
