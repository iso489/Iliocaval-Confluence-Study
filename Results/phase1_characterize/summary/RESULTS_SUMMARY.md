# Summary tables: results

Produced by `Code/phase1_characterize/summary/summary_tables.py` from the case-level outputs in `Results/phase1_characterize/`; inputs, hashes, versions and seeds are in `run_record.json`. Percentages are rounded plainly to one decimal and intervals print through `statlib.format_ci`.

## 1. Paired change, inference grid minus native grid (`t1_paired_projection.csv`)

- Measurable frame: 203/1,873 = +10.8 percentage points (95% CI: 9.4, 12.3; 1,759 participants, participant-block percentile).
- Discordant examinations: 205 disconnected only on the inference grid, 2 disconnected only on the native grid.
- Clustered McNemar (Durkalski 2003, participants as clusters): X² = 189.1, P = 4.9e-43; exact unclustered McNemar P = 2.1e-58. Both P < .001.
- Conservative sensitivity (26 empty-label examinations counted as disconnected, 1,899 processed): native 48.3% (95% CI: 46.1, 50.7), inference 59.0% (95% CI: 56.8, 61.3), paired change +10.7 points (95% CI: 9.3, 12.1).

- spacing_le_1p5mm: 0/276 = +0.0 points (95% CI: DEGENERATE_ZERO_WIDTH)
- spacing_gt1p5_to3mm: 142/1,145 = +12.4 points (95% CI: 10.5, 14.4)
- spacing_gt3mm: 61/452 = +13.5 points (95% CI: 10.4, 16.8)

## 2. Medians with participant-block percentile intervals (`t1_median_intervals.csv`)

- Table 1, median separation, mm, AMOS: 5.24 (95% CI: 5.05, 6.00)
- Table 1, median separation, mm, SAROS: 3.50 (95% CI: 3.00, 4.24)
- Table 1, median separation, mm, autoPET: 2.63 (95% CI: 2.50, 2.96)
- Table S3, median bottleneck radius, mm, native: 1.61 (95% CI: 1.56, 1.68)
- Table S3, median bottleneck radius, mm, inference: 1.50 (95% CI: DEGENERATE_ZERO_WIDTH)

## 3. Corridor volume as a percentage of the venous labels, disconnected examinations (`t1_corridor_fraction.csv`)

- native grid (n = 892): median 0.02% (IQR, 0.01-0.07); maximum 1.86%; median corridor 18 voxels (25.7 mm³) against median venous labels 109,935 mm³
- inference grid (n = 1,095): median 0.03% (IQR, 0.02-0.08); maximum 1.89%; median corridor 12 voxels (40.5 mm³) against median venous labels 115,142 mm³

## 4. Bottleneck at a fixed 1.5 mm, connected examinations (`t1_bottleneck_fixed.csv`)

- native grid, bottleneck_radius<=min_spacing (published row): 299/981 (30.5%; 95% CI: 27.6, 33.4)
- native grid, bottleneck_radius<=1.5mm: 445/981 (45.4%; 95% CI: 42.2, 48.6)
- inference grid, bottleneck_radius<=min_spacing (published row): 721/778 (92.7%; 95% CI: 90.8, 94.4)
- inference grid, bottleneck_radius<=1.5mm: 721/778 (92.7%; 95% CI: 90.8, 94.4)

## 5. Labels in disconnected corridors (`t1_corridor_labels.csv`, `t1_corridor_label_frequency.csv`)

- native grid, corridor_contains_nonvascular_label: 94/892 (10.5%; 95% CI: 8.6, 12.6)
- native grid, corridor_contains_other_vascular_label: 0/892 (0.0%; 95% CI: DEGENERATE_ZERO_WIDTH)
- inference grid, corridor_contains_nonvascular_label: 90/1,095 (8.2%; 95% CI: 6.6, 9.9)
- inference grid, corridor_contains_other_vascular_label: 0/1,095 (0.0%; 95% CI: DEGENERATE_ZERO_WIDTH)
- native grid, arterial corridors (77 of 892) by label (`t1_corridor_arterial_labels.csv`): aorta 7; iliac_artery_left 14; iliac_artery_right 59
- inference grid, arterial corridors (73 of 1,095) by label (`t1_corridor_arterial_labels.csv`): aorta 7; iliac_artery_left 12; iliac_artery_right 57
- native grid, most frequent: vertebrae_L4 67; vertebrae_L5 45; vertebrae_L3 30; vertebrae_L2 7; liver 5; small_bowel 3; colon 2; iliopsoas_left 2
- inference grid, most frequent: vertebrae_L4 59; vertebrae_L5 36; vertebrae_L3 29; vertebrae_L2 8; liver 6; iliopsoas_left 3; small_bowel 3; colon 2
- Corridor location, native grid: 10 of 892 corridors contain an abdominal organ label; corridor centroid minus right iliac cranial end, median 5.0 mm (10th-90th percentile, -1.2 to 35.0)
- Corridor location, inference grid: 14 of 1,095 corridors contain an abdominal organ label; corridor centroid minus right iliac cranial end, median 3.3 mm (10th-90th percentile, -1.7 to 29.1)

## 6. SAROS disconnection by region (`t1_saros_region.csv`)

- all (392 processed): 177/366 (48.4%; 95% CI: 43.3, 53.5; Wilson)
- abdomen (270 processed): 83/244 (34.0%; 95% CI: 28.4, 40.2; Wilson)
- wholebody (122 processed): 94/122 (77.0%; 95% CI: 68.8, 83.6; Wilson)

## 7. autoPET disconnection by contrast description, all spacings (`t1_contrast_metadata.csv`)

- all: 381/1,014 (37.6%; 95% CI: 34.4, 40.8; participant-block percentile)
- with: 348/953 (36.5%; 95% CI: 33.3, 39.8; participant-block percentile)
- without: 33/61 (54.1%; 95% CI: 39.3, 69.0; participant-block percentile)

## 8. Age by sex, Table 1 participant rule (`t1_age_by_sex.csv`)

- AMOS, female (n = 184): median 54.5 years (IQR, 43.0-66.25); mean 54.6; range 21-94
- AMOS, male (n = 309): median 55.0 years (IQR, 42.0-64.0); mean 53.9; range 19-93
- AMOS, all (n = 493): median 55.0 years (IQR, 43.0-65.0); mean 54.1; range 19-94
- autoPET, female (n = 396): median 60.0 years (IQR, 50.0-71.0); mean 58.9; range 15-95
- autoPET, male (n = 500): median 62.0 years (IQR, 52.0-72.0); mean 60.5; range 11-92
- autoPET, all (n = 900): median 62.0 years (IQR, 51.0-72.0); mean 59.7; range 11-95

## 9. Agreement between repeat autoPET examinations (`t1_repeat_examinations.csv`, Table S4)

- binary, agreement: 79.1 (95% CI: 70.7, 87.1); 81 participants, 195 examinations, 161 pairs
- binary, nominal_kappa: 0.56 (95% CI: 0.38, 0.73); 81 participants, 195 examinations, 161 pairs
- binary, nominal_ac1: 0.60 (95% CI: 0.43, 0.76); 81 participants, 195 examinations, 161 pairs
- three_class_nominal, agreement: 65.9 (95% CI: 56.3, 75.2); 81 participants, 195 examinations, 161 pairs
- three_class_nominal, nominal_kappa: 0.47 (95% CI: 0.32, 0.61); 81 participants, 195 examinations, 161 pairs
- three_class_nominal, nominal_ac1: 0.50 (95% CI: 0.36, 0.64); 81 participants, 195 examinations, 161 pairs

## 10. Corridors that submodel merging could have altered (`t1_merge_bound.csv`)

- native grid: 2 of 892 disconnected corridors (0.2%) hold a label written after the cardiac submodel (iliopsoas_left); 94 hold any non-vascular label
- inference grid: 3 of 1,095 disconnected corridors (0.3%) hold a label written after the cardiac submodel (iliopsoas_left); 90 hold any non-vascular label

## 11. Released annotations: adjacency and separation (`t1_reference_annotations.csv`)

- disconnected_26: 326/600 (54.3%; 95% CI: 50.3, 58.3; Wilson)
- disconnected_6: 361/600 (60.2%; 95% CI: 56.2, 64.0; Wilson)
- connected_26_disconnected_6: 35/274 (12.8%; 95% CI: 9.3, 17.2; Wilson)
- disconnected_26_boundary_excluded: 272/511 (53.2%; 95% CI: 48.9, 57.5; Wilson)
- disconnected_6_boundary_excluded: 301/511 (58.9%; 95% CI: 54.6, 63.1; Wilson)
- separation_le_1_slice_unit: 26/326 (8.0%; 95% CI: 5.5, 11.4; Wilson)
- separation_gt_2_slice_units: 268/326 (82.2%; 95% CI: 77.7, 86.0; Wilson)
- separation_mm_median: 6.36 mm (IQR, 4.24-11.95); n = 326
- nonmeasurable:left_iliac_label_only_absent: 377 of 628
- nonmeasurable:ivc_label_only_absent: 36 of 628
- nonmeasurable:both_labels_absent: 215 of 628
- nonmeasurable:head_neck_chest_or_heart_study: 393 of 628
- nonmeasurable:abdomen_pelvis_legs_or_whole_body_study: 161 of 628
- nonmeasurable:study_description_unplaced: 74 of 628

## 12. Paired change where projection can act, and under six-connectivity (`t1_projection_extended.csv`)

- 26-connectivity, all: 203/1,873 = +10.8 points (95% CI: 9.4, 12.3); closed 205, opened 2; clustered McNemar P = 4.9e-43
- 26-connectivity, spacing_gt1p5mm: 203/1,597 = +12.7 points (95% CI: 11.1, 14.4); closed 205, opened 2; clustered McNemar P = 3.8e-43
- 26-connectivity, spacing_le1p5mm: 0/276 = +0.0 points (95% CI: DEGENERATE_ZERO_WIDTH); closed 0, opened 0; clustered McNemar P = nan
- 6-connectivity, all: 382/1,873 = +20.4 points (95% CI: 18.5, 22.3); closed 390, opened 8; clustered McNemar P = 1.7e-77
- 6-connectivity, spacing_gt1p5mm: 382/1,597 = +23.9 points (95% CI: 21.7, 26.0); closed 390, opened 8; clustered McNemar P = 1.1e-77
- 6-connectivity, spacing_le1p5mm: 0/276 = +0.0 points (95% CI: DEGENERATE_ZERO_WIDTH); closed 0, opened 0; clustered McNemar P = nan

## 13. Disconnection by grid, adjacency and cohort (`t1_inference_grid_rates.csv`)

- native grid, 26-connectivity, pooled: 892/1,873 (47.6%; 95% CI: 45.4, 49.9; participant-block percentile)
- native grid, 26-connectivity, AMOS: 334/493 (67.7%; 95% CI: 63.5, 71.7; Wilson)
- native grid, 26-connectivity, SAROS: 177/366 (48.4%; 95% CI: 43.3, 53.5; Wilson)
- native grid, 26-connectivity, autoPET: 381/1,014 (37.6%; 95% CI: 34.4, 40.8; participant-block percentile)
- native grid, 6-connectivity, pooled: 1110/1,873 (59.3%; 95% CI: 57.0, 61.5; participant-block percentile)
- native grid, 6-connectivity, AMOS: 368/493 (74.6%; 95% CI: 70.6, 78.3; Wilson)
- native grid, 6-connectivity, SAROS: 205/366 (56.0%; 95% CI: 50.9, 61.0; Wilson)
- native grid, 6-connectivity, autoPET: 537/1,014 (53.0%; 95% CI: 49.7, 56.3; participant-block percentile)
- inference grid, 26-connectivity, pooled: 1095/1,873 (58.5%; 95% CI: 56.2, 60.7; participant-block percentile)
- inference grid, 26-connectivity, AMOS: 374/493 (75.9%; 95% CI: 71.9, 79.4; Wilson)
- inference grid, 26-connectivity, SAROS: 223/366 (60.9%; 95% CI: 55.8, 65.8; Wilson)
- inference grid, 26-connectivity, autoPET: 498/1,014 (49.1%; 95% CI: 45.8, 52.5; participant-block percentile)
- inference grid, 6-connectivity, pooled: 1492/1,873 (79.7%; 95% CI: 77.8, 81.6; participant-block percentile)
- inference grid, 6-connectivity, AMOS: 434/493 (88.0%; 95% CI: 84.9, 90.6; Wilson)
- inference grid, 6-connectivity, SAROS: 294/366 (80.3%; 95% CI: 75.9, 84.1; Wilson)
- inference grid, 6-connectivity, autoPET: 764/1,014 (75.3%; 95% CI: 72.5, 78.2; participant-block percentile)

## 14. Cohort-and-spacing model of disconnection by grid and adjacency (`t1_inference_grid_model.csv`)

- native grid, 26-connectivity, C(spacing_category)[T.gt1p5_le3]: odds ratio 0.568 (95% CI: 0.434, 0.744); P = 4.0e-05
- native grid, 26-connectivity, C(spacing_category)[T.gt3]: odds ratio 0.426 (95% CI: 0.296, 0.613); P = 4.3e-06
- native grid, 26-connectivity, C(cohort)[T.SAROS]: odds ratio 0.382 (95% CI: 0.286, 0.511); P = 1.2e-10
- native grid, 26-connectivity, C(cohort)[T.autoPET]: odds ratio 0.217 (95% CI: 0.161, 0.293); P = 6.0e-23
- inference grid, 26-connectivity, C(spacing_category)[T.gt1p5_le3]: odds ratio 0.980 (95% CI: 0.750, 1.282); P = 8.8e-01
- inference grid, 26-connectivity, C(spacing_category)[T.gt3]: odds ratio 0.978 (95% CI: 0.678, 1.412); P = 9.1e-01
- inference grid, 26-connectivity, C(cohort)[T.SAROS]: odds ratio 0.495 (95% CI: 0.365, 0.669); P = 5.5e-06
- inference grid, 26-connectivity, C(cohort)[T.autoPET]: odds ratio 0.306 (95% CI: 0.225, 0.416); P = 7.2e-14
- native grid, 6-connectivity, C(spacing_category)[T.gt1p5_le3]: odds ratio 0.322 (95% CI: 0.238, 0.436); P = 3.7e-13
- native grid, 6-connectivity, C(spacing_category)[T.gt3]: odds ratio 0.225 (95% CI: 0.152, 0.332); P = 1.3e-13
- native grid, 6-connectivity, C(cohort)[T.SAROS]: odds ratio 0.341 (95% CI: 0.251, 0.461); P = 5.0e-12
- native grid, 6-connectivity, C(cohort)[T.autoPET]: odds ratio 0.255 (95% CI: 0.185, 0.350); P = 6.0e-17
- inference grid, 6-connectivity, C(spacing_category)[T.gt1p5_le3]: odds ratio 1.086 (95% CI: 0.792, 1.489); P = 6.1e-01
- inference grid, 6-connectivity, C(spacing_category)[T.gt3]: odds ratio 0.924 (95% CI: 0.589, 1.450); P = 7.3e-01
- inference grid, 6-connectivity, C(cohort)[T.SAROS]: odds ratio 0.541 (95% CI: 0.365, 0.802); P = 2.3e-03
- inference grid, 6-connectivity, C(cohort)[T.autoPET]: odds ratio 0.380 (95% CI: 0.251, 0.574); P = 4.6e-06

## 15. Participants under 18 and autoPET diagnosis per study (`t1_population_sensitivity.csv`)

- table1_subject_record:LUNG_CANCER: 169
- table1_subject_record:LYMPHOMA: 146
- table1_subject_record:MELANOMA: 191
- table1_subject_record:NEGATIVE: 508
- diagnosis_per_study:LUNG_CANCER: 51/168 (30.4%; 95% CI: 23.9, 37.7; Wilson)
- diagnosis_per_study:LYMPHOMA: 73/145 (50.3%; 95% CI: 42.1, 58.3; participant-block percentile)
- diagnosis_per_study:MELANOMA: 59/188 (31.4%; 95% CI: 24.2, 38.8; participant-block percentile)
- diagnosis_per_study:NEGATIVE: 198/513 (38.6%; 95% CI: 34.0, 43.1; participant-block percentile)
- under_18:autoPET: 4 examinations, 4 participants, 3 disconnected
- without_under_18:autoPET: 378/1,010 (37.4%; 95% CI: 34.2, 40.7; participant-block percentile)
- without_under_18:pooled: 889/1,869 (47.6%; 95% CI: 45.3, 49.9; participant-block percentile)

## 16. SAROS source collections and autoPET (`t1_cross_dataset_overlap.csv`)

- saros_collection:ACRIN-FLT-Breast: 32
- saros_collection:ACRIN-NSCLC-FDG-PET: 48
- saros_collection:Anti-PD-1_Lung: 11
- saros_collection:C4KC-KiTS: 171
- saros_collection:COVID-19-NY-SBU: 1
- saros_collection:CPTAC-CM: 1
- saros_collection:CPTAC-LSCC: 3
- saros_collection:CPTAC-LUAD: 1
- saros_collection:CPTAC-PDA: 8
- saros_collection:CPTAC-UCEC: 26
- saros_collection:Lung-PET-CT-Dx: 2
- saros_collection:NSCLC Radiogenomics: 7
- saros_collection:NSCLC-Radiomics-Genomics: 2
- saros_collection:Pancreas-CT: 34
- saros_collection:Soft-tissue-Sarcoma: 6
- saros_collection:TCGA-LIHC: 31
- saros_collection:TCGA-LUAD: 2
- saros_collection:TCGA-LUSC: 3
- saros_collection:TCGA-STAD: 2
- saros_collection:TCGA-UCEC: 1
- saros_collections: 20
- saros_cases_in_autopet_collection: 0
- study_uid_shared_with_autopet: 0
- series_uid_shared_with_autopet: 0

## 17. Comparators beside the left iliac endpoint, disconnected examinations (`t1_comparators.csv`)

- native grid, 26-connectivity, ivc_lciv, pooled: 892/1,873 (47.6%; 95% CI: 45.4, 49.9; participant-block percentile)
- native grid, 26-connectivity, ivc_lciv, AMOS: 334/493 (67.7%; 95% CI: 63.5, 71.7; Wilson)
- native grid, 26-connectivity, ivc_lciv, SAROS: 177/366 (48.4%; 95% CI: 43.3, 53.5; Wilson)
- native grid, 26-connectivity, ivc_lciv, autoPET: 381/1,014 (37.6%; 95% CI: 34.4, 40.8; participant-block percentile)
- native grid, 26-connectivity, ivc_lciv_not_direct, pooled: 1489/1,873 (79.5%; 95% CI: 77.6, 81.4; participant-block percentile)
- native grid, 26-connectivity, ivc_lciv_not_direct, AMOS: 429/493 (87.0%; 95% CI: 83.8, 89.7; Wilson)
- native grid, 26-connectivity, ivc_lciv_not_direct, SAROS: 277/366 (75.7%; 95% CI: 71.0, 79.8; Wilson)
- native grid, 26-connectivity, ivc_lciv_not_direct, autoPET: 783/1,014 (77.2%; 95% CI: 74.5, 80.0; participant-block percentile)
- native grid, 26-connectivity, ivc_rciv, pooled: 865/1,871 (46.2%; 95% CI: 43.9, 48.6; participant-block percentile)
- native grid, 26-connectivity, ivc_rciv, AMOS: 309/493 (62.7%; 95% CI: 58.3, 66.8; Wilson)
- native grid, 26-connectivity, ivc_rciv, SAROS: 177/364 (48.6%; 95% CI: 43.5, 53.7; Wilson)
- native grid, 26-connectivity, ivc_rciv, autoPET: 379/1,014 (37.4%; 95% CI: 34.2, 40.6; participant-block percentile)
- native grid, 26-connectivity, aorta_iliac_artery_left, pooled: 130/1,873 (6.9%; 95% CI: 5.8, 8.1; participant-block percentile)
- native grid, 26-connectivity, aorta_iliac_artery_left, AMOS: 74/493 (15.0%; 95% CI: 12.1, 18.4; Wilson)
- native grid, 26-connectivity, aorta_iliac_artery_left, SAROS: 34/366 (9.3%; 95% CI: 6.7, 12.7; Wilson)
- native grid, 26-connectivity, aorta_iliac_artery_left, autoPET: 22/1,014 (2.2%; 95% CI: 1.3, 3.2; participant-block percentile)
- native grid, 26-connectivity, aorta_iliac_artery_right, pooled: 114/1,873 (6.1%; 95% CI: 5.0, 7.2; participant-block percentile)
- native grid, 26-connectivity, aorta_iliac_artery_right, AMOS: 52/493 (10.5%; 95% CI: 8.1, 13.6; Wilson)
- native grid, 26-connectivity, aorta_iliac_artery_right, SAROS: 32/366 (8.7%; 95% CI: 6.3, 12.1; Wilson)
- native grid, 26-connectivity, aorta_iliac_artery_right, autoPET: 30/1,014 (3.0%; 95% CI: 1.9, 4.1; participant-block percentile)
- native grid, 6-connectivity, ivc_lciv, pooled: 1110/1,873 (59.3%; 95% CI: 57.0, 61.5; participant-block percentile)
- native grid, 6-connectivity, ivc_lciv, AMOS: 368/493 (74.6%; 95% CI: 70.6, 78.3; Wilson)
- native grid, 6-connectivity, ivc_lciv, SAROS: 205/366 (56.0%; 95% CI: 50.9, 61.0; Wilson)
- native grid, 6-connectivity, ivc_lciv, autoPET: 537/1,014 (53.0%; 95% CI: 49.7, 56.3; participant-block percentile)
- native grid, 6-connectivity, ivc_lciv_not_direct, pooled: 1655/1,873 (88.4%; 95% CI: 86.8, 89.9; participant-block percentile)
- native grid, 6-connectivity, ivc_lciv_not_direct, AMOS: 454/493 (92.1%; 95% CI: 89.4, 94.2; Wilson)
- native grid, 6-connectivity, ivc_lciv_not_direct, SAROS: 312/366 (85.2%; 95% CI: 81.2, 88.5; Wilson)
- native grid, 6-connectivity, ivc_lciv_not_direct, autoPET: 889/1,014 (87.7%; 95% CI: 85.4, 89.9; participant-block percentile)
- native grid, 6-connectivity, ivc_rciv, pooled: 1044/1,871 (55.8%; 95% CI: 53.5, 58.1; participant-block percentile)
- native grid, 6-connectivity, ivc_rciv, AMOS: 326/493 (66.1%; 95% CI: 61.8, 70.2; Wilson)
- native grid, 6-connectivity, ivc_rciv, SAROS: 204/364 (56.0%; 95% CI: 50.9, 61.1; Wilson)
- native grid, 6-connectivity, ivc_rciv, autoPET: 514/1,014 (50.7%; 95% CI: 47.4, 54.1; participant-block percentile)
- native grid, 6-connectivity, aorta_iliac_artery_left, pooled: 142/1,873 (7.6%; 95% CI: 6.4, 8.8; participant-block percentile)
- native grid, 6-connectivity, aorta_iliac_artery_left, AMOS: 81/493 (16.4%; 95% CI: 13.4, 20.0; Wilson)
- native grid, 6-connectivity, aorta_iliac_artery_left, SAROS: 39/366 (10.7%; 95% CI: 7.9, 14.2; Wilson)
- native grid, 6-connectivity, aorta_iliac_artery_left, autoPET: 22/1,014 (2.2%; 95% CI: 1.3, 3.2; participant-block percentile)
- native grid, 6-connectivity, aorta_iliac_artery_right, pooled: 118/1,873 (6.3%; 95% CI: 5.2, 7.4; participant-block percentile)
- native grid, 6-connectivity, aorta_iliac_artery_right, AMOS: 54/493 (11.0%; 95% CI: 8.5, 14.0; Wilson)
- native grid, 6-connectivity, aorta_iliac_artery_right, SAROS: 34/366 (9.3%; 95% CI: 6.7, 12.7; Wilson)
- native grid, 6-connectivity, aorta_iliac_artery_right, autoPET: 30/1,014 (3.0%; 95% CI: 1.9, 4.1; participant-block percentile)
- inference grid, 26-connectivity, ivc_lciv, pooled: 1095/1,873 (58.5%; 95% CI: 56.2, 60.7; participant-block percentile)
- inference grid, 26-connectivity, ivc_lciv, AMOS: 374/493 (75.9%; 95% CI: 71.9, 79.4; Wilson)
- inference grid, 26-connectivity, ivc_lciv, SAROS: 223/366 (60.9%; 95% CI: 55.8, 65.8; Wilson)
- inference grid, 26-connectivity, ivc_lciv, autoPET: 498/1,014 (49.1%; 95% CI: 45.8, 52.5; participant-block percentile)
- inference grid, 26-connectivity, ivc_lciv_not_direct, pooled: 1622/1,873 (86.6%; 95% CI: 85.0, 88.2; participant-block percentile)
- inference grid, 26-connectivity, ivc_lciv_not_direct, AMOS: 455/493 (92.3%; 95% CI: 89.6, 94.3; Wilson)
- inference grid, 26-connectivity, ivc_lciv_not_direct, SAROS: 317/366 (86.6%; 95% CI: 82.7, 89.7; Wilson)
- inference grid, 26-connectivity, ivc_lciv_not_direct, autoPET: 850/1,014 (83.8%; 95% CI: 81.4, 86.2; participant-block percentile)
- inference grid, 26-connectivity, ivc_rciv, pooled: 1107/1,871 (59.2%; 95% CI: 56.9, 61.4; participant-block percentile)
- inference grid, 26-connectivity, ivc_rciv, AMOS: 366/493 (74.2%; 95% CI: 70.2, 77.9; Wilson)
- inference grid, 26-connectivity, ivc_rciv, SAROS: 227/364 (62.4%; 95% CI: 57.3, 67.2; Wilson)
- inference grid, 26-connectivity, ivc_rciv, autoPET: 514/1,014 (50.7%; 95% CI: 47.3, 54.1; participant-block percentile)
- inference grid, 26-connectivity, aorta_iliac_artery_left, pooled: 139/1,873 (7.4%; 95% CI: 6.3, 8.6; participant-block percentile)
- inference grid, 26-connectivity, aorta_iliac_artery_left, AMOS: 79/493 (16.0%; 95% CI: 13.1, 19.5; Wilson)
- inference grid, 26-connectivity, aorta_iliac_artery_left, SAROS: 38/366 (10.4%; 95% CI: 7.7, 13.9; Wilson)
- inference grid, 26-connectivity, aorta_iliac_artery_left, autoPET: 22/1,014 (2.2%; 95% CI: 1.3, 3.2; participant-block percentile)
- inference grid, 26-connectivity, aorta_iliac_artery_right, pooled: 119/1,873 (6.4%; 95% CI: 5.3, 7.5; participant-block percentile)
- inference grid, 26-connectivity, aorta_iliac_artery_right, AMOS: 59/493 (12.0%; 95% CI: 9.4, 15.1; Wilson)
- inference grid, 26-connectivity, aorta_iliac_artery_right, SAROS: 31/366 (8.5%; 95% CI: 6.0, 11.8; Wilson)
- inference grid, 26-connectivity, aorta_iliac_artery_right, autoPET: 29/1,014 (2.9%; 95% CI: 1.9, 4.0; participant-block percentile)
- inference grid, 6-connectivity, ivc_lciv, pooled: 1492/1,873 (79.7%; 95% CI: 77.8, 81.6; participant-block percentile)
- inference grid, 6-connectivity, ivc_lciv, AMOS: 434/493 (88.0%; 95% CI: 84.9, 90.6; Wilson)
- inference grid, 6-connectivity, ivc_lciv, SAROS: 294/366 (80.3%; 95% CI: 75.9, 84.1; Wilson)
- inference grid, 6-connectivity, ivc_lciv, autoPET: 764/1,014 (75.3%; 95% CI: 72.5, 78.2; participant-block percentile)
- inference grid, 6-connectivity, ivc_lciv_not_direct, pooled: 1781/1,873 (95.1%; 95% CI: 94.1, 96.0; participant-block percentile)
- inference grid, 6-connectivity, ivc_lciv_not_direct, AMOS: 475/493 (96.3%; 95% CI: 94.3, 97.7; Wilson)
- inference grid, 6-connectivity, ivc_lciv_not_direct, SAROS: 348/366 (95.1%; 95% CI: 92.4, 96.9; Wilson)
- inference grid, 6-connectivity, ivc_lciv_not_direct, autoPET: 958/1,014 (94.5%; 95% CI: 93.0, 95.8; participant-block percentile)
- inference grid, 6-connectivity, ivc_rciv, pooled: 1508/1,871 (80.6%; 95% CI: 78.7, 82.4; participant-block percentile)
- inference grid, 6-connectivity, ivc_rciv, AMOS: 432/493 (87.6%; 95% CI: 84.4, 90.2; Wilson)
- inference grid, 6-connectivity, ivc_rciv, SAROS: 299/364 (82.1%; 95% CI: 77.9, 85.7; Wilson)
- inference grid, 6-connectivity, ivc_rciv, autoPET: 777/1,014 (76.6%; 95% CI: 73.8, 79.4; participant-block percentile)
- inference grid, 6-connectivity, aorta_iliac_artery_left, pooled: 156/1,873 (8.3%; 95% CI: 7.2, 9.6; participant-block percentile)
- inference grid, 6-connectivity, aorta_iliac_artery_left, AMOS: 90/493 (18.3%; 95% CI: 15.1, 21.9; Wilson)
- inference grid, 6-connectivity, aorta_iliac_artery_left, SAROS: 42/366 (11.5%; 95% CI: 8.6, 15.1; Wilson)
- inference grid, 6-connectivity, aorta_iliac_artery_left, autoPET: 24/1,014 (2.4%; 95% CI: 1.5, 3.4; participant-block percentile)
- inference grid, 6-connectivity, aorta_iliac_artery_right, pooled: 125/1,873 (6.7%; 95% CI: 5.6, 7.8; participant-block percentile)
- inference grid, 6-connectivity, aorta_iliac_artery_right, AMOS: 63/493 (12.8%; 95% CI: 10.1, 16.0; Wilson)
- inference grid, 6-connectivity, aorta_iliac_artery_right, SAROS: 33/366 (9.0%; 95% CI: 6.5, 12.4; Wilson)
- inference grid, 6-connectivity, aorta_iliac_artery_right, autoPET: 29/1,014 (2.9%; 95% CI: 1.9, 4.0; participant-block percentile)

## 18. IVC to right iliac vein within each left iliac class (`t1_comparator_crosstab.csv`)

- native grid, 26-connectivity, direct: right iliac connected in 323
- native grid, 26-connectivity, direct: right iliac disconnected in 60
- native grid, 26-connectivity, direct: right iliac nonmeasurable in 1
- native grid, 26-connectivity, disconnected: right iliac connected in 92
- native grid, 26-connectivity, disconnected: right iliac disconnected in 799
- native grid, 26-connectivity, disconnected: right iliac nonmeasurable in 1
- native grid, 26-connectivity, right_iliac_mediated: right iliac connected in 591
- native grid, 26-connectivity, right_iliac_mediated: right iliac disconnected in 6
- native grid, 26-connectivity, right_iliac_mediated: right iliac nonmeasurable in 0
- native grid, 6-connectivity, direct: right iliac connected in 172
- native grid, 6-connectivity, direct: right iliac disconnected in 46
- native grid, 6-connectivity, direct: right iliac nonmeasurable in 0
- native grid, 6-connectivity, disconnected: right iliac connected in 117
- native grid, 6-connectivity, disconnected: right iliac disconnected in 991
- native grid, 6-connectivity, disconnected: right iliac nonmeasurable in 2
- native grid, 6-connectivity, right_iliac_mediated: right iliac connected in 538
- native grid, 6-connectivity, right_iliac_mediated: right iliac disconnected in 7
- native grid, 6-connectivity, right_iliac_mediated: right iliac nonmeasurable in 0
- inference grid, 26-connectivity, direct: right iliac connected in 190
- inference grid, 26-connectivity, direct: right iliac disconnected in 60
- inference grid, 26-connectivity, direct: right iliac nonmeasurable in 1
- inference grid, 26-connectivity, disconnected: right iliac connected in 54
- inference grid, 26-connectivity, disconnected: right iliac disconnected in 1040
- inference grid, 26-connectivity, disconnected: right iliac nonmeasurable in 1
- inference grid, 26-connectivity, right_iliac_mediated: right iliac connected in 520
- inference grid, 26-connectivity, right_iliac_mediated: right iliac disconnected in 7
- inference grid, 26-connectivity, right_iliac_mediated: right iliac nonmeasurable in 0
- inference grid, 6-connectivity, direct: right iliac connected in 45
- inference grid, 6-connectivity, direct: right iliac disconnected in 47
- inference grid, 6-connectivity, direct: right iliac nonmeasurable in 0
- inference grid, 6-connectivity, disconnected: right iliac connected in 37
- inference grid, 6-connectivity, disconnected: right iliac disconnected in 1453
- inference grid, 6-connectivity, disconnected: right iliac nonmeasurable in 2
- inference grid, 6-connectivity, right_iliac_mediated: right iliac connected in 281
- inference grid, 6-connectivity, right_iliac_mediated: right iliac disconnected in 8
- inference grid, 6-connectivity, right_iliac_mediated: right iliac nonmeasurable in 0

## 19. Repeat pairs that disagree, and edge or corner contacts (`t1_repeat_contacts.csv`)

- native grid: 34 of 161 pairs disagree; the connected examination had a face-connected path in 24 and needed an edge or corner contact in 10 (all connections: 218 of 981)
- inference grid: 36 of 161 pairs disagree; the connected examination had a face-connected path in 11 and needed an edge or corner contact in 25 (all connections: 397 of 778)

## 20. Empty labels in the nonmeasurable examinations (`t1_nonmeasurable_labels.csv`)

- pooled: 26 of 1899 nonmeasurable; left iliac label empty only 26, IVC only 0, both 0; left and right iliac labels both empty 24
- AMOS: 0 of 493 nonmeasurable; left iliac label empty only 0, IVC only 0, both 0; left and right iliac labels both empty 0
- SAROS: 26 of 392 nonmeasurable; left iliac label empty only 26, IVC only 0, both 0; left and right iliac labels both empty 24
- autoPET: 0 of 1014 nonmeasurable; left iliac label empty only 0, IVC only 0, both 0; left and right iliac labels both empty 0

## 21. Betti numbers of the venous union, (1, 0, 0) the anatomical topology (`t1_betti.csv`)

- predictions, all, anatomical_topology, pooled: 156/1,873 (8.3%; 95% CI: 7.1, 9.6; participant-block percentile)
- predictions, all, anatomical_topology, AMOS: 46/493 (9.3%; 95% CI: 7.1, 12.2; Wilson)
- predictions, all, anatomical_topology, SAROS: 37/366 (10.1%; 95% CI: 7.4, 13.6; Wilson)
- predictions, all, anatomical_topology, autoPET: 73/1,014 (7.2%; 95% CI: 5.6, 8.9; participant-block percentile)
- predictions, all, b0_gt1, pooled: 977/1,873 (52.2%; 95% CI: 49.9, 54.5; participant-block percentile)
- predictions, all, b0_gt1, AMOS: 363/493 (73.6%; 95% CI: 69.6, 77.3; Wilson)
- predictions, all, b0_gt1, SAROS: 189/366 (51.6%; 95% CI: 46.5, 56.7; Wilson)
- predictions, all, b0_gt1, autoPET: 425/1,014 (41.9%; 95% CI: 38.7, 45.3; participant-block percentile)
- predictions, all, b1_gt0, pooled: 989/1,873 (52.8%; 95% CI: 50.5, 55.0; participant-block percentile)
- predictions, all, b1_gt0, AMOS: 172/493 (34.9%; 95% CI: 30.8, 39.2; Wilson)
- predictions, all, b1_gt0, SAROS: 175/366 (47.8%; 95% CI: 42.7, 52.9; Wilson)
- predictions, all, b1_gt0, autoPET: 642/1,014 (63.3%; 95% CI: 60.3, 66.4; participant-block percentile)
- predictions, all, b2_gt0, pooled: 346/1,873 (18.5%; 95% CI: 16.7, 20.3; participant-block percentile)
- predictions, all, b2_gt0, AMOS: 31/493 (6.3%; 95% CI: 4.5, 8.8; Wilson)
- predictions, all, b2_gt0, SAROS: 76/366 (20.8%; 95% CI: 16.9, 25.2; Wilson)
- predictions, all, b2_gt0, autoPET: 239/1,014 (23.6%; 95% CI: 20.8, 26.4; participant-block percentile)
- predictions, connected, anatomical_topology, pooled: 156/981 (15.9%; 95% CI: 13.6, 18.2; participant-block percentile)
- predictions, connected, anatomical_topology, AMOS: 46/159 (28.9%; 95% CI: 22.4, 36.4; Wilson)
- predictions, connected, anatomical_topology, SAROS: 37/189 (19.6%; 95% CI: 14.5, 25.8; Wilson)
- predictions, connected, anatomical_topology, autoPET: 73/633 (11.5%; 95% CI: 9.0, 14.2; participant-block percentile)
- predictions, connected, b0_gt1, pooled: 85/981 (8.7%; 95% CI: 7.0, 10.5; participant-block percentile)
- predictions, connected, b0_gt1, AMOS: 29/159 (18.2%; 95% CI: 13.0, 25.0; Wilson)
- predictions, connected, b0_gt1, SAROS: 12/189 (6.3%; 95% CI: 3.7, 10.8; Wilson)
- predictions, connected, b0_gt1, autoPET: 44/633 (7.0%; 95% CI: 5.1, 9.1; participant-block percentile)
- predictions, connected, b1_gt0, pooled: 730/981 (74.4%; 95% CI: 71.7, 77.1; participant-block percentile)
- predictions, connected, b1_gt0, AMOS: 90/159 (56.6%; 95% CI: 48.8, 64.1; Wilson)
- predictions, connected, b1_gt0, SAROS: 131/189 (69.3%; 95% CI: 62.4, 75.4; Wilson)
- predictions, connected, b1_gt0, autoPET: 509/633 (80.4%; 95% CI: 77.2, 83.5; participant-block percentile)
- predictions, connected, b2_gt0, pooled: 273/981 (27.8%; 95% CI: 24.9, 30.8; participant-block percentile)
- predictions, connected, b2_gt0, AMOS: 20/159 (12.6%; 95% CI: 8.3, 18.6; Wilson)
- predictions, connected, b2_gt0, SAROS: 56/189 (29.6%; 95% CI: 23.6, 36.5; Wilson)
- predictions, connected, b2_gt0, autoPET: 197/633 (31.1%; 95% CI: 27.3, 35.0; participant-block percentile)
- released_annotations, all, anatomical_topology, all: 96/600 (16.0%; 95% CI: 13.3, 19.1; Wilson)
- released_annotations, all, b0_gt1, all: 368/600 (61.3%; 95% CI: 57.4, 65.1; Wilson)
- released_annotations, all, b1_gt0, all: 195/600 (32.5%; 95% CI: 28.9, 36.3; Wilson)
- released_annotations, all, b2_gt0, all: 43/600 (7.2%; 95% CI: 5.4, 9.5; Wilson)

## 22. Vertebral level of the corridor, disconnected examinations (`t1_gap_level.csv`)

- pooled, bin L1: 1 of 872 with a valid ruler (892 disconnected)
- pooled, bin L2: 13 of 872 with a valid ruler (892 disconnected)
- pooled, bin L3: 52 of 872 with a valid ruler (892 disconnected)
- pooled, bin L4: 454 of 872 with a valid ruler (892 disconnected)
- pooled, bin L5: 335 of 872 with a valid ruler (892 disconnected)
- pooled, bin S1: 17 of 872 with a valid ruler (892 disconnected)
- pooled, label L1/L2: 2 of 872 with a valid ruler (892 disconnected)
- pooled, label L2: 7 of 872 with a valid ruler (892 disconnected)
- pooled, label L2/L3: 10 of 872 with a valid ruler (892 disconnected)
- pooled, label L3: 29 of 872 with a valid ruler (892 disconnected)
- pooled, label L3/L4: 66 of 872 with a valid ruler (892 disconnected)
- pooled, label L4: 222 of 872 with a valid ruler (892 disconnected)
- pooled, label L4/L5: 354 of 872 with a valid ruler (892 disconnected)
- pooled, label L5: 153 of 872 with a valid ruler (892 disconnected)
- pooled, label L5/S1: 19 of 872 with a valid ruler (892 disconnected)
- pooled, label S1: 10 of 872 with a valid ruler (892 disconnected)
- pooled, median_level 4.39: 872 of 872 with a valid ruler (892 disconnected)
- AMOS, bin L2: 10 of 332 with a valid ruler (334 disconnected)
- AMOS, bin L3: 24 of 332 with a valid ruler (334 disconnected)
- AMOS, bin L4: 193 of 332 with a valid ruler (334 disconnected)
- AMOS, bin L5: 93 of 332 with a valid ruler (334 disconnected)
- AMOS, bin S1: 12 of 332 with a valid ruler (334 disconnected)
- AMOS, label L1/L2: 1 of 332 with a valid ruler (334 disconnected)
- AMOS, label L2: 4 of 332 with a valid ruler (334 disconnected)
- AMOS, label L2/L3: 7 of 332 with a valid ruler (334 disconnected)
- AMOS, label L3: 17 of 332 with a valid ruler (334 disconnected)
- AMOS, label L3/L4: 25 of 332 with a valid ruler (334 disconnected)
- AMOS, label L4: 104 of 332 with a valid ruler (334 disconnected)
- AMOS, label L4/L5: 120 of 332 with a valid ruler (334 disconnected)
- AMOS, label L5: 38 of 332 with a valid ruler (334 disconnected)
- AMOS, label L5/S1: 10 of 332 with a valid ruler (334 disconnected)
- AMOS, label S1: 6 of 332 with a valid ruler (334 disconnected)
- AMOS, median_level 4.28: 332 of 332 with a valid ruler (334 disconnected)
- SAROS, bin L2: 1 of 170 with a valid ruler (177 disconnected)
- SAROS, bin L3: 11 of 170 with a valid ruler (177 disconnected)
- SAROS, bin L4: 86 of 170 with a valid ruler (177 disconnected)
- SAROS, bin L5: 71 of 170 with a valid ruler (177 disconnected)
- SAROS, bin S1: 1 of 170 with a valid ruler (177 disconnected)
- SAROS, label L2: 1 of 170 with a valid ruler (177 disconnected)
- SAROS, label L3: 4 of 170 with a valid ruler (177 disconnected)
- SAROS, label L3/L4: 20 of 170 with a valid ruler (177 disconnected)
- SAROS, label L4: 44 of 170 with a valid ruler (177 disconnected)
- SAROS, label L4/L5: 62 of 170 with a valid ruler (177 disconnected)
- SAROS, label L5: 38 of 170 with a valid ruler (177 disconnected)
- SAROS, label L5/S1: 1 of 170 with a valid ruler (177 disconnected)
- SAROS, median_level 4.38: 170 of 170 with a valid ruler (177 disconnected)
- autoPET, bin L1: 1 of 370 with a valid ruler (381 disconnected)
- autoPET, bin L2: 2 of 370 with a valid ruler (381 disconnected)
- autoPET, bin L3: 17 of 370 with a valid ruler (381 disconnected)
- autoPET, bin L4: 175 of 370 with a valid ruler (381 disconnected)
- autoPET, bin L5: 171 of 370 with a valid ruler (381 disconnected)
- autoPET, bin S1: 4 of 370 with a valid ruler (381 disconnected)
- autoPET, label L1/L2: 1 of 370 with a valid ruler (381 disconnected)
- autoPET, label L2: 2 of 370 with a valid ruler (381 disconnected)
- autoPET, label L2/L3: 3 of 370 with a valid ruler (381 disconnected)
- autoPET, label L3: 8 of 370 with a valid ruler (381 disconnected)
- autoPET, label L3/L4: 21 of 370 with a valid ruler (381 disconnected)
- autoPET, label L4: 74 of 370 with a valid ruler (381 disconnected)
- autoPET, label L4/L5: 172 of 370 with a valid ruler (381 disconnected)
- autoPET, label L5: 77 of 370 with a valid ruler (381 disconnected)
- autoPET, label L5/S1: 8 of 370 with a valid ruler (381 disconnected)
- autoPET, label S1: 4 of 370 with a valid ruler (381 disconnected)
- autoPET, median_level 4.46: 370 of 370 with a valid ruler (381 disconnected)

## 23. Disconnection by quartile of whole-label venous attenuation (`t1_attenuation_quartiles.csv`)

- pooled, quartile 1: 327/457 (71.6%; 95% CI: 67.3, 75.5; Wilson)
- pooled, quartile 2: 192/465 (41.3%; 95% CI: 36.9, 45.8; Wilson)
- pooled, quartile 3: 186/476 (39.1%; 95% CI: 34.8, 43.5; Wilson)
- pooled, quartile 4: 187/475 (39.4%; 95% CI: 35.1, 43.8; Wilson)
- AMOS, quartile 1: 199/273 (72.9%; 95% CI: 67.3, 77.8; Wilson)
- AMOS, quartile 2: 68/95 (71.6%; 95% CI: 61.8, 79.7; Wilson)
- AMOS, quartile 3: 22/47 (46.8%; 95% CI: 33.3, 60.8; Wilson)
- AMOS, quartile 4: 45/78 (57.7%; 95% CI: 46.6, 68.0; Wilson)
- SAROS, quartile 1: 95/123 (77.2%; 95% CI: 69.1, 83.8; Wilson)
- SAROS, quartile 2: 36/118 (30.5%; 95% CI: 22.9, 39.3; Wilson)
- SAROS, quartile 3: 27/81 (33.3%; 95% CI: 24.0, 44.1; Wilson)
- SAROS, quartile 4: 19/44 (43.2%; 95% CI: 29.7, 57.8; Wilson)
- autoPET, quartile 1: 33/61 (54.1%; 95% CI: 41.7, 66.0; Wilson)
- autoPET, quartile 2: 88/252 (34.9%; 95% CI: 29.3, 41.0; Wilson)
- autoPET, quartile 3: 137/348 (39.4%; 95% CI: 34.4, 44.6; Wilson)
- autoPET, quartile 4: 123/353 (34.8%; 95% CI: 30.1, 40.0; Wilson)
- lowest quartile against the other three, cohort-stratified Mantel-Haenszel odds ratio 2.65 (95% CI: 2.05, 3.42; P = 1.8e-14)
