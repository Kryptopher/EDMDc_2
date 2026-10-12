# Provenance

Every file in this repository, where it came from, and its SHA-256. Paths in the *Origin* column are
relative to the authors' research repository (`quadcopter_EDMD/Sanjay/EDMDC_MPC`).
`src/` files are byte-identical to the versions recorded in the paper's frozen protocols.

The files under `provenance/` are unmodified originals kept for audit. They therefore still contain the
original local paths of the machine on which the experiments were run. Nothing in `src/`, `experiments/`
or `configs/` reads or depends on them.

| File | Origin | Role | SHA-256 |
|---|---|---|---|
| `.gitattributes` | written for this release | release code/documentation | `0bf74143e24cf413eab053ed1d24b76f2faff41328b61f9130dabb0d8817eeab` |
| `.gitignore` | written for this release | release code/documentation | `069642b4b5d8ce0ad2e5cf8eefdce2b96b57cd7318db4a847ebdaadae2aef311` |
| `CITATION.cff` | written for this release | release code/documentation | `79feb280bc4f57ec98e5b8b4b692eff89754915710a6f94c4003e7eebd2f657d` |
| `configs/edmdc_training.json` | artifacts/wrench_revisit_2026-09-21/training_v2/training_status.json | config; edited for release: fields added for the published .npz archive and the model content digest | `b97567fd9b0971e1adbbf343b1445d342dba7b13c6fb80f69183b0ef051fc75e` |
| `configs/interception_candidates.json` | wrench_interception_v1/candidates.json | config | `d33f65fdc25819ace37db3d05b6ab3ab29c243de37e417ee5322c79f082eb221` |
| `configs/interception_controllers.json` | overnight_campaign_20260929/continuation_20260930/interception/selection.json | config | `db962973aa11c533cc041985958ab59693e696bdac3740affcd4c0f876879297` |
| `configs/interception_scenarios.json` | artifacts/target_information_ablation_20261002/TARGET_INFORMATION_ABLATION_PROTOCOL.json | config | `d8f11e675759d82e7e78e5110608d810b1885a264e5d5ae3b13a4abea23e0dbf` |
| `configs/selection_scenarios.json` | wrench_interception_v1/protocol.json | config | `de5a3e372c877abb66d0dcaf0e6b3e403ecd2b9f29d1d7bc8c8cfc9bcc9b13ad` |
| `configs/timing_cases.json` | artifacts/wrench_controller_tuning_v1/timing_pass/tasks.json | config | `07f1ab61f23dc9d88f7721e5da28ae72419b5fb07055a638d2b4c5a0d7a75227` |
| `configs/tracking_controllers.json` | artifacts/wrench_controller_tuning_v1/selected_configs | config | `66d565bad316562aa4278d3ee86e774844a176297eaf600c5eb39140ee7cf81e` |
| `data/models/wrench_physics42_dt001.pkl` | artifacts/wrench_revisit_2026-09-21/training_v2/models/wrench_physics42_dt001.pkl | EDMDc model used in the paper | `090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4` |
| `DATA.md` | written for this release | release code/documentation | `09e02d0752b41ab71ea87fc77f25356fe2cf3d83f65344c96cb3b7a9aed80442` |
| `experiments/_common.py` | written for this release | release code/documentation | `27197b067f0c3251c66b682907d2672de2b36ef8f1f5e747c9b619b36f51b7d7` |
| `experiments/_train_launcher.py` | written for this release | release code/documentation | `bca7a583eb4f7097fcca737c0b41b11d23266b51de5a50e71d4f929561dcfb29` |
| `experiments/analyze_interception.py` | written for this release | release code/documentation | `41fe2ddcec595947cc5dee14cdc41d8314f800b85bcc187c9e0f16d6a389b75c` |
| `experiments/analyze_model.py` | written for this release | release code/documentation | `9bc64dda3197a1e8a99dcf99e886794a840aeb526b7c200f3a70c6ce515cf671` |
| `experiments/analyze_selection.py` | written for this release | release code/documentation | `d89d94a59aac80d215aff9df2ad80fe113ebfe7397aa493adc9bb0a3f6240bfe` |
| `experiments/analyze_timing.py` | written for this release | release code/documentation | `d43746f34caa3f9547d196dd8c99010bdb6ccebf45a12f0c99440a8189554807` |
| `experiments/analyze_tracking.py` | written for this release | release code/documentation | `2eac8b56234cc135a5f49ede48d348238a3e8bd0d565f6772240eb51db3a9ae5` |
| `experiments/dataset_io.py` | written for this release | release code/documentation | `a65f5c120a5e5b039207c8dca6269bf2e3bf666bd111b5e912625078ed971460` |
| `experiments/evaluate_prediction.py` | written for this release | release code/documentation | `5080cb1a769d84b05b5be015043fb386aaaa861056f807ce7e600abeaec9fb58` |
| `experiments/interception.py` | written for this release | release code/documentation | `f44b3f775cef2aad08f3b63177c4a7844cfd1a71bf810a818f318e2b3132acbd` |
| `experiments/make_claim_table.py` | written for this release | release code/documentation | `4ecfcd6bad3d3da3386619345c8fa1851fffcb570a9fcae0f7f7bca334e01245` |
| `experiments/model_digest.py` | written for this release | release code/documentation | `60a23791e5cd8274e3e9e88707caf7d30c4fa707013cd1830402ae4e1f154e88` |
| `experiments/selection.py` | written for this release | release code/documentation | `126bc0ff9aaea8bb4f78fde61c916f3fc4a26f9e731299b09d27e39bbaf490a0` |
| `experiments/timing.py` | written for this release | release code/documentation | `c895fb1418ae73230ecf5641aa65fee1f3fb25b0dae7dc412b205261320698ea` |
| `experiments/tracking.py` | written for this release | release code/documentation | `5621c17b45061a70c7f9985f8d7167f8bf8571c8c6813126380dd95adb7b6d63` |
| `experiments/train_edmdc.py` | written for this release | release code/documentation | `1429f935bc610ce3c30e79a754a4cceb0dc14378974c89745e872b037ce49364` |
| `figures/fig2_selected_cases.csv` | generated by the figure script in this repository | figure output | `11a2d01e3df2c6808122dbbcf6bac938af25103118de66294e188eb6e4e54485` |
| `figures/fig2_tracking.eps` | generated by the figure script in this repository | figure output | `5e1193d8bd868f18aeea61e6d1d769bf05c0745ad361bf9a0630604791e488b3` |
| `figures/fig2_tracking.pdf` | generated by the figure script in this repository | figure output | `48371a8819b1f9d6dca08ed47fb992857207dd0c3b249c0380ecde3a194b58ba` |
| `figures/fig2_tracking.png` | generated by the figure script in this repository | figure output | `99be265419b6a893099b43b3430e2734828cf6b039b44b6bc9772eee559bee6c` |
| `figures/fig2_tracking.py` | paper_acc_revision/final_figures/plot_tracking_matched_final.py | figure script (patched paths); edited for release: unused provenance-writer function removed | `3bc83815765108dca2d49781a1fac9f56878bc1b5de2458cd3e773e2ca21000a` |
| `figures/fig2_tracking.svg` | generated by the figure script in this repository | figure output | `489aa0bcbcacb2b7cdc52a33e8040bb75ca0715827a4930de332d15b5bb74704` |
| `figures/fig3_timing.py` | paper_acc_revision/final_figures/plot_3_controller_update_timing_ecdf.py | figure script (patched paths) | `0e3b296d14a3ab78ee18217107fc09bc6a80a8fa48e442c6680064a71310eb12` |
| `figures/fig3_timing_ecdf.eps` | generated by the figure script in this repository | figure output | `ee7af393177d694cf163321120cf27bf0c6ad22ff46096dd5ebd4492cc7265ad` |
| `figures/fig3_timing_ecdf.pdf` | generated by the figure script in this repository | figure output | `57e73b8932163bb7780101ab9052beccf9557e217e56f58875c24c6d607ece35` |
| `figures/fig3_timing_ecdf.png` | generated by the figure script in this repository | figure output | `6ada3f76033c547cbbaba3cb6fc423cf209d2b3f21b1c8f2c21e497a192f9d4e` |
| `figures/fig3_timing_ecdf.svg` | generated by the figure script in this repository | figure output | `8e971a56a01df041df2bbc67dbb031d8dc7a12d3b75d9e5bd9aa590afb871f81` |
| `figures/fig3_timing_ecdf_full.eps` | generated by the figure script in this repository | figure output | `fff65160499e0acb1c9124ad21c810c78a9c22f0ee6c3034022ffa54f7b3692a` |
| `figures/fig3_timing_ecdf_full.pdf` | generated by the figure script in this repository | figure output | `861326dc40a01053de2a0b55155207466f9597eb8d56c5880ac89eb70cf34f3b` |
| `figures/fig3_timing_ecdf_full.png` | generated by the figure script in this repository | figure output | `8427b2f32bade4a587115aef3e31c444d580b23b80abb559b9501bedf0894063` |
| `figures/fig3_timing_ecdf_full.svg` | generated by the figure script in this repository | figure output | `df719a26a8e1461c36d895c6be79e339e402d2b7affb64bd3ab6aa7661b90a97` |
| `figures/fig4_interception.eps` | generated by the figure script in this repository | figure output | `281d0c2b47b1b99dc52766a89093f9f800c8cec2b6245aab8988837623cfc88a` |
| `figures/fig4_interception.pdf` | generated by the figure script in this repository | figure output | `3a3b47824bf384f2ebb85e1712e9fa0d6d0ff5f7a9b4d9f9c5b92d84ea2ff95f` |
| `figures/fig4_interception.png` | generated by the figure script in this repository | figure output | `9c20eee360d1faf01df329c0dab5925a261784bbe698738495e1a1e098a85119` |
| `figures/fig4_interception.py` | paper_acc_revision/final_figures/plot_interception_current_cv_simplified.py | figure script (patched paths); edited for release: leftover print of a manuscript path removed | `cab45fe6286396fbe14772867062b5f6d4da5520fd9ccba60150c94e1deb371a` |
| `figures/fig4_interception.svg` | generated by the figure script in this repository | figure output | `089fe6d26b61b92204e0c0d6f830f82e1442b0a28c87f4539fbb5c9092ff0663` |
| `figures/paper/1_UAV_sketch.eps` | submitted figure bundle ACC_2027_EDMD.zip, Figures_R1/1_UAV_sketch.eps (identical in Figures_R0) | Fig. 1 as in paper (drawing, no generating script) | `4358c5dca42b7b8df111a7be514251e23428bf993f85ad1b2ba9bd7e09480cbc` |
| `figures/paper/2_3D_Trajectory_and_Errors.eps` | paper_acc_revision/final_figures/tracking_matched_3d_errors.eps | figure as in paper | `6ca4c473644002ae6daa29385febe77eca9a40702c908ddfadd49ec59dee18a3` |
| `figures/paper/3_controller_update_timing_ecdf.eps` | paper_acc_revision/Figures_R1/3_controller_update_timing_ecdf.eps | figure as in paper | `657790d18ceff19bb2211d7daa564c74f1a4ca2c8489bfe0e73d2cefe44d2f47` |
| `figures/paper/4_Interception.eps` | paper_acc_revision/Figures_R1/4_Interception.eps | figure as in paper | `551552ba5bbc1a0063fa9151f26cdbccf6bb43c295e795a463b228f99ccfd2b4` |
| `LICENSE` | written for this release | release code/documentation | `b25a17785ad6bdc0bb2f3941ab4395fec66d3f7be5549832f9a707ff6332ba45` |
| `provenance/figure_png_reference/3_controller_update_timing_ecdf.png` | paper_acc_revision/final_figures/3_controller_update_timing_ecdf.png | reference rendering for pixel comparison | `6ada3f76033c547cbbaba3cb6fc423cf209d2b3f21b1c8f2c21e497a192f9d4e` |
| `provenance/figure_png_reference/interception_current_cv_simplified.png` | paper_acc_revision/final_figures/interception_current_cv_simplified.png | reference rendering for pixel comparison | `9c20eee360d1faf01df329c0dab5925a261784bbe698738495e1a1e098a85119` |
| `provenance/figure_png_reference/tracking_matched_3d_errors.png` | paper_acc_revision/final_figures/tracking_matched_3d_errors.png | reference rendering for pixel comparison | `99be265419b6a893099b43b3430e2734828cf6b039b44b6bc9772eee559bee6c` |
| `provenance/original/artifacts/acc_balanced_high_tilt_v1/runs_high_tilt_attitude_n60.json` | artifacts/acc_balanced_high_tilt_v1/runs_high_tilt_attitude_n60.json | provenance: high-tilt augmentation metadata | `5cb5c882f1286931bcc99225e5f65fbd44edb3d46eb32a91b5dac6f6de5730b5` |
| `provenance/original/artifacts/acc_balanced_high_tilt_v1/runs_mixed_n360_high_tilt.json` | artifacts/acc_balanced_high_tilt_v1/runs_mixed_n360_high_tilt.json | provenance: training dataset metadata | `c16690f1180d7c42e2fbbbb2cbb2002dbc4d7f63e5e3bdef2545523c95836109` |
| `provenance/original/artifacts/interception_timing_repeat_20261002/run_timing.py` | artifacts/interception_timing_repeat_20261002/run_timing.py | provenance: four-cell timing runner | `96af8ef02c4f3ab56a874a92892945c688defbddaeb3ab24cfa706d0a4a58215` |
| `provenance/original/artifacts/interception_timing_repeat_20261002/summarize_four_cells.py` | artifacts/interception_timing_repeat_20261002/summarize_four_cells.py | provenance: four-cell timing summary | `70a3e3c23d028b2f4f9e3d354cc21466b01bced838353604f8dcc69f2cf1ee37` |
| `provenance/original/artifacts/interception_timing_repeat_20261002/TIMING_PROTOCOL.json` | artifacts/interception_timing_repeat_20261002/TIMING_PROTOCOL.json | provenance: four-cell timing protocol | `c797b4a4b5c5c2e854be6e00a959b1a85e4e336c009b0eb61086907128d54a9b` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/analyze.py` | artifacts/target_information_ablation_20261002/scripts/analyze.py | provenance: Current/CV ablation: analysis | `dd9c4e6ad1d58093285df8c132fd24f77312d89de4753ae49dfa681f9b753510` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/finalize.py` | artifacts/target_information_ablation_20261002/scripts/finalize.py | provenance: Current/CV ablation: finalize | `e2115f60f353ef0bce09056d7aefe6880d1feb8b50e86b3200773dd93dd1ee34` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/prepare.py` | artifacts/target_information_ablation_20261002/scripts/prepare.py | provenance: Current/CV ablation: preparation | `724fb86332fb89c6bdfdf894f92f25e20ab2d5f3248877757b0b1fc0641a47a0` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/run.py` | artifacts/target_information_ablation_20261002/scripts/run.py | provenance: Current/CV ablation: runner | `349d02413d82c2d9981b271733b735c1e7aa0570c9a50e0957ff31c77a1ff722` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/runtime.py` | artifacts/target_information_ablation_20261002/scripts/runtime.py | provenance: Current/CV ablation: episode runtime | `6ba496b5e24f259257429922354a3876896ef2b3d3af1af497c9aa946eb3980b` |
| `provenance/original/artifacts/target_information_ablation_20261002/scripts/validate.py` | artifacts/target_information_ablation_20261002/scripts/validate.py | provenance: Current/CV ablation: validation | `4e5f3960c079cd3bfe46ed9771153d28195cf5edc3dbef4171b6d90f37a50492` |
| `provenance/original/artifacts/target_information_ablation_20261002/TARGET_INFORMATION_ABLATION_PROTOCOL.json` | artifacts/target_information_ablation_20261002/TARGET_INFORMATION_ABLATION_PROTOCOL.json | provenance: Current/CV ablation protocol | `1c3a849a47be90cd6ea46105d90b4ae68bb4cebb59f18bac2afde1979dd4deae` |
| `provenance/original/artifacts/wrench_controller_tuning_v1/amendment_01.json` | artifacts/wrench_controller_tuning_v1/amendment_01.json | provenance: tracking timing-pass amendment | `be65e63212a3821a4b823fe2ac119f104f597341ca33a7a9adc20dd872471421` |
| `provenance/original/artifacts/wrench_controller_tuning_v1/wrench_tuning_protocol.json` | artifacts/wrench_controller_tuning_v1/wrench_tuning_protocol.json | provenance: tracking protocol | `796d60e4b06b72290191fd048922a70e63fe9f278d9a70f8a652f718fbe4a821` |
| `provenance/original/artifacts/wrench_revisit_2026-09-21/training_v2/selected_model.json` | artifacts/wrench_revisit_2026-09-21/training_v2/selected_model.json | provenance: selected model record | `b841f6e9279e3a6881e99049e08d8d7d864046c62eb8c69af6a0ed99e50f977e` |
| `provenance/original/artifacts/wrench_revisit_2026-09-21/training_v2/training_physics42.log` | artifacts/wrench_revisit_2026-09-21/training_v2/training_physics42.log | provenance: training log (lambda grid, transitions) | `6deda30eca2947ee20b85a33f7ecb10dc3ece80e994a1016c6ba6dd4e49fbb94` |
| `provenance/original/artifacts/wrench_revisit_2026-09-21/training_v2/training_status.json` | artifacts/wrench_revisit_2026-09-21/training_v2/training_status.json | provenance: training contract/status | `23ee7b06a8ae5bc79fb1fc258e7160d25e78a4c4870b0aa10f741ee62f8a83d6` |
| `provenance/original/evaluate_wrench_revisit.py` | evaluate_wrench_revisit.py | provenance: held-out rolling prediction check | `c96451f32c06ed82cf29bdebb74211aab5957b5bb30c2bf0fd16b218f0472c07` |
| `provenance/original/overnight_campaign_20260929/continuation_20260930/interception/selection.json` | overnight_campaign_20260929/continuation_20260930/interception/selection.json | provenance: frozen selection (A=c04, B=c22) | `2a922609747f9e38d83280c6ca4b611f4a5d1f32898de52b8ac6bf8ca41de8f9` |
| `provenance/original/overnight_campaign_20260929/continuation_20260930/interception/selection.sha256.json` | overnight_campaign_20260929/continuation_20260930/interception/selection.sha256.json | provenance: selection hash | `2b3b3cb0a2cad8537fdafe60dc69f1e0195303430b9e268366e67070eb9479b7` |
| `provenance/original/overnight_campaign_20260929/continuation_20260930/recover.py` | overnight_campaign_20260929/continuation_20260930/recover.py | provenance: selection campaign recovery/continuation | `967e345dd8379145e54d8d4e5b9f84aab3d536590f052b7f020fba134da61e10` |
| `provenance/original/paper_gap_audit_20261001/analyze.py` | paper_gap_audit_20261001/analyze.py | provenance: crossed oracle study: analysis | `5e8a4bfa2ddb4b555f41e49044c45f635fe3a049ff18c9cdfc66f702a04bf9a4` |
| `provenance/original/paper_gap_audit_20261001/prepare.py` | paper_gap_audit_20261001/prepare.py | provenance: crossed oracle study: preparation | `e6dff2fd7a792cc84ca89b73baecc4c2f1287fd010a32bf366806d9314b29dc9` |
| `provenance/original/paper_gap_audit_20261001/PROTOCOL.json` | paper_gap_audit_20261001/PROTOCOL.json | provenance: crossed oracle study protocol | `962ce04f79bd23ea17c1461288169c3adb6e9ff991cdcea2fda4d456d054e80a` |
| `provenance/original/paper_gap_audit_20261001/run.py` | paper_gap_audit_20261001/run.py | provenance: crossed oracle study: runner | `6d8545396590fa4b9af84de9c9d60d481e9de57bcae6d697d42ec9186cbd2009` |
| `provenance/original/paper_gap_audit_20261001/runtime.py` | paper_gap_audit_20261001/runtime.py | provenance: crossed oracle study: episode runtime | `ea33ea9b0ab893a34868f3bd8c3bba50315155fdeef58c4903fdab9fb2b7329c` |
| `provenance/original/run_wrench_revisit_training.py` | run_wrench_revisit_training.py | provenance: EDMDc training wrapper used for the paper model | `c6dbd6c4d9b5401c08b071badb080cb625f44154dadd5d09f6cd4f1feae37f91` |
| `provenance/original/wrench_interception_v1/campaign.py` | wrench_interception_v1/campaign.py | provenance: interception controller selection campaign (screening/validation) | `c78f6571991280c087ea3040ac4749e56bf1d29c67a2a3200053ccac4a368aab` |
| `provenance/original/wrench_interception_v1/candidates.json` | wrench_interception_v1/candidates.json | provenance: 25 LHS candidates (all arms) | `417916c12aef60fce2c758bdcbaecd9eadd7ced4fa5af8c0f9535d148f32a3cc` |
| `provenance/original/wrench_interception_v1/protocol.json` | wrench_interception_v1/protocol.json | provenance: selection campaign protocol | `11cdc6018c89753b1d769519916dbf3423078382a48c896c107ba2fd93c72a52` |
| `README.md` | written for this release | release code/documentation | `38bec299b94480909cd90c20612628d89319aebf40f84a41c36b5b1f53ccf575` |
| `requirements.txt` | written for this release | release code/documentation | `84d1f34f04c75b0fbe6663447b9585b88569211b2d8e3b5ba48beb12b05558ad` |
| `results/figure_inputs/interception/EDMDc-B_cv_3103001001.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_EDMDc-B_cv_3103001001/trace.npz | figure input (Fig. 4) | `bdfad206dcb3de2d9ba94b346d2a89debd69023ad0e7b7ad65fd3666c2c67937` |
| `results/figure_inputs/interception/EDMDc-B_cv_3103002006.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_EDMDc-B_cv_3103002006/trace.npz | figure input (Fig. 4) | `87f866bc25f982456676030b92b7550c0690f21606e7d180041557b2bc001313` |
| `results/figure_inputs/interception/EDMDc-B_cv_3103003001.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_EDMDc-B_cv_3103003001/trace.npz | figure input (Fig. 4) | `a16612b92b43dcb1cd4b61cec2bb7ce5b520486cbfb27a83b5024a9649d07578` |
| `results/figure_inputs/interception/Linear-A_cv_3103001001.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_Linear-A_cv_3103001001/trace.npz | figure input (Fig. 4) | `6a8a9deed5fcb0aa2e79d582c4d775ea5803de8378156f86b5c55275bde324c5` |
| `results/figure_inputs/interception/Linear-A_cv_3103002006.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_Linear-A_cv_3103002006/trace.npz | figure input (Fig. 4) | `7b733c73da07cb29fa8926a3e8cdf778bf34dd771e56444ae345b05263145bc7` |
| `results/figure_inputs/interception/Linear-A_cv_3103003001.npz` | artifacts/target_information_ablation_20261002/episodes/primary/primary_Linear-A_cv_3103003001/trace.npz | figure input (Fig. 4) | `458268d9ddf329f17bd805bdff2389623935d54b08011cad08c0412efaf76c36` |
| `results/figure_inputs/tracking/W1/figure8_92001041_scale1.75_path.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W1/c23/traces/figure8_92001041_scale1.75_path/yaw_linear.npz | figure input (Fig. 2) | `02fad7421b667dc692567a49c3e723a9e9091d8437fd0b4515a97f4be1ac53a7` |
| `results/figure_inputs/tracking/W1/helix_92000003_scale1.75_fixed.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W1/c23/traces/helix_92000003_scale1.75_fixed/yaw_linear.npz | figure input (Fig. 2) | `c7e6cf3f4b198868679b988c73afecdb240cd5033668af7b30cdc102c4a4a1d0` |
| `results/figure_inputs/tracking/W1/lissajous_92002049_scale1.75_path.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W1/c23/traces/lissajous_92002049_scale1.75_path/yaw_linear.npz | figure input (Fig. 2) | `36021fcef31f82e7503fb1e11b060415a6f186fdf7929b026103baf6eb3e7987` |
| `results/figure_inputs/tracking/W2a/figure8_92001041_scale1.75_path.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W2a/c23/traces/figure8_92001041_scale1.75_path/edmdc.npz | figure input (Fig. 2) | `dacd7d53341612fb2867c8baf54b120ca8fa71ca64f0d3174e6789b312bd31f9` |
| `results/figure_inputs/tracking/W2a/helix_92000003_scale1.75_fixed.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W2a/c23/traces/helix_92000003_scale1.75_fixed/edmdc.npz | figure input (Fig. 2) | `c702e5b57252ca7a24414ce37cac521cfb6c89e01b92a3e1723327545a24b720` |
| `results/figure_inputs/tracking/W2a/lissajous_92002049_scale1.75_path.npz` | artifacts/wrench_controller_tuning_v1/blind_test/W2a/c23/traces/lissajous_92002049_scale1.75_path/edmdc.npz | figure input (Fig. 2) | `04c98dafb193acbf93cfc2ad41ad9db1153862f97960209633b0b23d67e2182a` |
| `results/interception/episodes.json` | paper_gap_audit_20261001/C_EPISODES.json + artifacts/target_information_ablation_20261002/{primary,secondary}_EPISODES.json | frozen result: 640 interception episodes (4 cells x 4 information modes x 40 scenarios) | `9dee39e40736179ac10766d5eeb7805e636332ac805f5b39706c82414240d59b` |
| `results/interception/original_analysis_output.json` | artifacts/target_information_ablation_20261002/TARGET_INFORMATION_ABLATION_RESULTS.json | frozen result: original analysis output; edited for release: absolute paths replaced by repo-relative "original:" paths | `bd5d829fdd851207be218cf086d83ecf539119a9a18b22b8dee3d55f96644bed` |
| `results/prediction/edmdc_validation_test_rolling_aggregate.csv` | artifacts/wrench_revisit_2026-09-21/prediction_evaluation/rolling_aggregate.csv | frozen result: EDMDc validation/test rolling RMSE | `c237e20c5a950dcca0f532e9f74ad7a6283516be436c8b046d0d85c1b47938ca` |
| `results/prediction/heldout_rolling_aggregate.csv` | artifacts/wrench_reidentification_2026-09-21/dataset_v1/prediction_evaluation/rolling_aggregate.csv | frozen result: held-out open-loop RMSE; edited for release: filtered to the paper model and the hover-linear baseline on the 9 held-out runs (rows of an unrelated re-identified model removed); labels renamed | `49d9af0950298666688e2d7a84c3bedccdb9d7a2063981f9562b8c9c83977be5` |
| `results/selection/finalists.json` | overnight_campaign_20260929\continuation_20260930\interception\validation\finalists.json | frozen result: selection | `b07f4180b15ec9d35de1d37acaef5441b2c0290cf36d0dd42f9a58ef57170d42` |
| `results/selection/screening_episodes.json` | wrench_interception_v1/screening/episodes.json | frozen result: screening (hover-linear W1, EDMDc W2a) | `ec57ec3ed999a5b91894f1a1065135252d374954bc9f23bf693bd71426230eb7` |
| `results/selection/screening_ranking.json` | overnight_campaign_20260929\continuation_20260930\interception\screening\ranking.json | frozen result: selection | `14bc9cc6c54ad60ad4cd551e160a338e32da96e29527e891170f7117464ed391` |
| `results/selection/validation_episodes.json` | overnight_campaign_20260929/continuation_20260930/interception/validation/episodes.json | frozen result: validation | `0a61c9d33733614722e32f2350fb85161ee6b178a1f814debf4a7a2e0a3cd656` |
| `results/selection/validation_ranking.json` | overnight_campaign_20260929\continuation_20260930\interception\validation\ranking.json | frozen result: selection | `99963e3be436d8f7f6e58e7eb3ce9cb829373e8946fd7f6a4bc137d81d74f4b6` |
| `results/timing/EDMDc-A_raw_update_times_ms.npz` | artifacts/interception_timing_repeat_20261002/EDMDc-A_raw_update_times_ms.npz | frozen result: raw update times (Fig. 3) | `89a9b5912d8dd6d61c32535656c7240e5f018dc1337346a88120bf23fe2a3f4a` |
| `results/timing/EDMDc-B_raw_update_times_ms.npz` | artifacts/interception_timing_repeat_20261002/EDMDc-B_raw_update_times_ms.npz | frozen result: raw update times (Fig. 3) | `7665db340d84629a8792c768a4fb7fe392b58a7af3797aade4f78c316f3e3b35` |
| `results/timing/four_cell_timing_results.json` | artifacts\interception_timing_repeat_20261002\FOUR_CELL_TIMING_RESULTS.json | frozen result: four-cell timing | `b2aaeb365f93040db1559e2b3cb7948592206694c63b3bf42bee25f434fcd472` |
| `results/timing/Linear-A_raw_update_times_ms.npz` | artifacts/interception_timing_repeat_20261002/Linear-A_raw_update_times_ms.npz | frozen result: raw update times (Fig. 3) | `d55f7f6364fac916b6803b21efa28f33d758d2b45a374ce883379b2077edbfd3` |
| `results/timing/Linear-B_raw_update_times_ms.npz` | artifacts/interception_timing_repeat_20261002/Linear-B_raw_update_times_ms.npz | frozen result: raw update times (Fig. 3) | `ab3f97c5ca883d7c2107fe07ea0564fe842c8ab3e0d5812cd1a73ce90db778a7` |
| `results/timing/matched_tracking_timing_summary.json` | artifacts/wrench_controller_tuning_v1/analysis/isolated_timing_summary.json | frozen result: matched tracking timing; edited for release: row of an unrelated model arm (W2b) removed | `f38aae88ab0b4236c9bc5301248e5380711de1241f24de69b34a88d9208970fb` |
| `results/timing/Trajectory-linear_matched_raw_update_times_ms.npz` | artifacts/wrench_controller_tuning_v1/timing_pass/W3 traces (solve_seconds x 1000) | frozen result: trajectory-linear raw update times (Fig. 3) | `ff4f07db48b1d8aa744a68bc27d2040f1c511f593a695741b85ee34135e8c4e1` |
| `results/tracking/blind_test_episodes.json` | artifacts/wrench_controller_tuning_v1/blind_test/episodes.json | frozen result: 200-scenario matched tracking (W1, W2a at c23; W3 at c11) | `01759cf63f93ad4af68e1cb16a73c8320d0a68e1016edba634b5a30bb9b8d12a` |
| `results/training/training_physics42.log` | artifacts/wrench_revisit_2026-09-21/training_v2/training_physics42.log | training log read by analyze_model.py; one absolute plot-output path replaced by "<research repository>" | `f017fce30b26bcdf9f9285d690884ae965de5cd81e66e3ab957e14ab7e5ddedd` |
| `src/append_high_tilt_augmentation.py` | append_high_tilt_augmentation.py | core library (byte-identical) | `24758642b7d8ac22ddf8a7f24d7f355eee241b2dceab2db71024e99f3dce3942` |
| `src/Cascaded_Controllers.py` | Cascaded_Controllers.py | core library (byte-identical) | `032890d773512b5b452b032d009800d7a872ce5c230a84d65348d320fd566a72` |
| `src/Closed_loop.py` | Closed_loop.py | core library (byte-identical) | `1dbfbf1735561bb4421ef4b2402459073be921c0a2462d2753a74000b78da112` |
| `src/compare_mpc.py` | compare_mpc.py | core library (byte-identical) | `e3a2e310ef6f5648568c3343a5eb2ab5cf2d8da148538c3d7eac5fe2677ea52c` |
| `src/compare_three.py` | compare_three.py | core library (byte-identical) | `f78dd285dbd30518ba48df3aa0b28e793b08c03adea7831d8444a0ecf0d2ba43` |
| `src/edmdc_mpc.py` | edmdc_mpc.py | core library (byte-identical) | `a9884524e39dded05c2738b55e0c28fe21578d5f4c63bf03a13978654568a4f6` |
| `src/edmdc_rollout_batch.py` | edmdc_rollout_batch.py | core library (byte-identical) | `a2566c82ff64f43a11533db9ed1eff6b3fa8e6202a30893b823ad5e3e2a55d1a` |
| `src/EDMDc_training.py` | EDMDc_training.py | core library (byte-identical) | `c579f25c09d583b62a697ec903bb21249734865f8e52fad1119e87c78aa9cba4` |
| `src/generate_high_tilt_augmentation.py` | generate_high_tilt_augmentation.py | core library (byte-identical) | `82028073b1da8f8080299168bd7e40d26eee629617ce9a1ca99584ed008453ee` |
| `src/Helperfcts.py` | Helperfcts.py | core library (byte-identical) | `f5be150f868c4e497d8cab83ec11874720b445e24278ff9ec448ddf9f7f82527` |
| `src/iPID_Trajectory_Controller.py` | iPID_Trajectory_Controller.py | core library (byte-identical) | `f9869ec9a885b02589134a5f5cbd4665e029a34410f91e4f8755953640413184` |
| `src/mix_traj.py` | mix_traj.py | core library (byte-identical) | `6e901f53030293ef524a480cbac191e17fbd4bec16ad167c9bf36f7ab1835d58` |
| `src/parallel_sim.py` | parallel_sim.py | core library (byte-identical) | `9f84f0bfa16d26b4347f44abcf6b1540491a2bfb39e4d1b0e9a8a63938090160` |
| `src/PID_Mixer.py` | PID_Mixer.py | core library (byte-identical) | `67648a4c96f0fec930ba381b30c4cd1da29f30f33a61993aead97e26b0de1a80` |
| `src/PID_Trajectory_Controller.py` | PID_Trajectory_Controller.py | core library (byte-identical) | `d579cae740d16a170c2563cd17708547d3c7df02bbec6604eb4ef1d9ec3e70cd` |
| `src/publication_scenarios.py` | publication_scenarios.py | core library (byte-identical) | `fb4479ee0cfae44def50d26ab460f179a29c728feed317b66d34bcd3cdbd7427` |
| `src/quadcopter.py` | quadcopter.py | core library (byte-identical) | `120469050255c5f0e45b0c0486bb13b9325d04619129098a7f9e03178862858f` |
| `src/replace_waypoint_family.py` | replace_waypoint_family.py | core library (byte-identical) | `5920f6df98baf41a1726c5d9ee854219aeec21af9f0e9b2615803d0825afb396` |
| `src/run_wrench_controller_tuning.py` | run_wrench_controller_tuning.py | core library (byte-identical) | `d8d1d08ede29b83978b2c808024134ac5e936e9b286357519146603edf8d2f04` |
| `src/run_wrench_revisit.py` | run_wrench_revisit.py | core library (byte-identical) | `a65769c599c9a665947b1bad8fb707396a80d1831b46dedbfbf7d36854336933` |
| `src/Simulation.py` | Simulation.py | core library (byte-identical) | `d89f139ad885a1467f5df075f5542319caed8c45f5a10409e2f6fb528ad2d72c` |
| `src/wrench_interception_v1/core.py` | wrench_interception_v1/core.py | core library (byte-identical) | `46005124f520a4c31f183c46c7db512bd370c085445389fc2ee1f64a82159ba5` |
| `src/wrench_interception_v1/targets.py` | wrench_interception_v1/targets.py | core library (byte-identical) | `5773d88745b8565fe1b5c647d34fc34752a52e3c7b1acaf4ee2d150871977350` |
| `src/wrench_revisit_mpc.py` | wrench_revisit_mpc.py | core library (byte-identical) | `880e8136042d57c3d19c0575ae51d6c2c41aa7ed606dc34a38ee19a41392aebb` |
| `tools/pkl_to_npz.py` | written for this release | release code/documentation | `09680069d5d0a5b6637d8632e16a70ce3cd0cf6322584aaed87e743c522f6c54` |
