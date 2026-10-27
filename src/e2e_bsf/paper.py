"""Global: scientific communication. Local: render an evidence-derived PDF with ReportLab.

Sources: https://docs.reportlab.com/reportlab/userguide/ch5_platypus/.
No numbers are invented; the paper receives the same results object as the explorer.
"""

from xml.sax.saxutils import escape  # Paragraph markup must not execute arbitrary artifact text.
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether


def write_paper(destination, evidence, markdown_table):
    """Local: coherent readable paper; global: keep scope, uncertainty and authorship explicit."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontSize=23, leading=28, spaceAfter=14))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 9
    story = []
    details = evidence["study_details"]  # Actual allocation, distinct from the configured epoch cap.
    by_condition = {row["condition"]: row for row in evidence["summaries"]}

    def paragraph(text, style="BodyText"):
        """Local: one shared text helper; global: consistent typesetting without duplicated layouts."""
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        """Create a modest section heading suitable for a technical report."""
        paragraph(text, "Heading2")

    def figure(filename, caption, width=6.8 * inch):
        """Scale a measured figure by its aspect ratio; never stretch scientific axes."""
        path = destination / "figures" / filename
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as opened:
                height = width * opened.height / opened.width
            story.append(KeepTogether([Image(str(path), width=width, height=height),
                                       Paragraph(caption, styles["SmallBody"])]))

    paragraph("End-to-End Block-Sparse Featurizers for a Frozen Vision Readout", "PaperTitle")
    paragraph("Libor Burian", "Heading2")
    paragraph(escape(evidence["label"]))
    heading("Abstract")
    paragraph("We study whether output-distribution matching can train a block-sparse representation that preserves a fixed vision classifier. "
              "A Grassmannian block-sparse featurizer replaces final DINOv3 patch activations. Its parameters are initialized randomly and "
              "optimized using forward KL divergence from the original composite classifier to the reconstructed classifier. "
              "Local reconstruction, scalar sparsity and a mixed objective provide comparisons. " + escape(evidence["narrative"]) +
              " These measurements concern the frozen ten-class readout; they do not establish semantic concept recovery.")
    heading("1. Context, motivation and related work")
    paragraph("Local sparse dictionary learning rewards activation reconstruction, whereas end-to-end sparse dictionary learning [1] "
              "rewards preservation of downstream output distributions. Block-sparse featurizers [2] group signed coordinates into "
              "low-dimensional charts, motivated by superposed manifolds. We combine the released Grassmannian implementation with "
              "the executable forward-KL convention of [1]. Prior work studies short KL fine-tuning [3]; this experiment instead "
              "trains every BSF condition from scratch. Later BSF analysis [4] motivates caution about splitting and stability.")
    paragraph("DINOv3 [5] produces embeddings. Consequently, the original model here is explicitly DINOv3 plus one independently fitted, "
              "then frozen, patch-only Imagenette classifier. The dataset and classifier are smaller adaptations of [2], not a "
              "reproduction of its complete ImageNet evaluation. The model was pretrained externally; held-out claims refer to the "
              "new featurizer and classifier procedure.")
    heading("2. Method")
    paragraph("Let H contain the final 196 patch vectors of width 768, excluding CLS and four register tokens. Fit a positional mean "
              "mu and scalar RMS s on training images. For x=(H-mu)/s, transpose each QR frame to obtain orthonormal rows D_g. The tied encoder computes "
              "z_g=gamma_g x D_g^T, retains the k blocks with largest Euclidean norms, and reconstructs x_hat=sum_g z_g D_g. "
              "Insert H_hat=s x_hat+mu without adding the reconstruction residual. The released code uses an unconstrained learned "
              "gain per block and differentiable QR every forward; these details differ from the paper's shared positive gain and "
              "periodic projection description. We reuse one QR result within each training forward and across batches within each frozen evaluation call [7].")
    paragraph("Pool each activation coordinate by its maximum absolute magnitude over patches. Fit a ten-class multinomial logistic "
              "readout on clean training activations, choosing C in {0.01,0.1,1,10} by development cross-entropy. Freeze the head for "
              "all conditions. With p=softmax(head(H)) and q=softmax(head(H_hat)), L_KL=mean_images sum_classes p(log p-log q). "
              "Teacher targets are detached, temperature is one, and log-softmax/KL use float32. Only the BSF is optimized.")
    paragraph("Main dictionary: 4096 blocks of width 3, keeping 16 blocks. The scalar ablation uses 12288 width-one blocks and 48 "
              "active coordinates. The smaller smoke configuration, if used in this report, is explicitly listed below. Scalar "
              "controls match total/active coordinates but have more gain parameters. They are tied signed featurizers, not ReLU SAEs. "
              "A mixed objective adds 0.1 times normalized patch MSE; the primary KL condition has no reconstruction penalty.")
    heading("3. Data, optimization and analysis")
    paragraph("Image SHA-256 and 64-bit perceptual hashes (Hamming distance at most four) define duplicate components before splitting. "
              "Official Imagenette training becomes an 80/20 grouped training/development split at seed 42; official validation is "
              "reserved for testing. Training members of test-related duplicate groups are excluded. Test groups inspected during "
              "the separate engineering smoke run are also excluded from the full study's test set. Normalization never refits on "
              "development or test. Perceptual hashing is a limited screen, not a guarantee that all related images are detected.")
    paragraph("Adam uses gradient clipping at one and cosine decay to 10% of the initial learning rate. The pilot compares 1e-4 and "
              "3e-4 equally, using independent seeds 0,1,2 on 510 training and 250 development images. A paired-difference power "
              "calculation targets 80% power for a 10% relative KL reduction, with 5-10 confirmatory seeds and at most 20 epochs. "
              "The measured time forecast includes a 25% overhead allowance. Every condition gets the same confirmatory image exposure. "
              "The 24-hour bound cannot be extended merely to obtain significance.")
    paragraph("Primary contrast is per-seed mean local-MSE KL minus KL-trained KL, paired by initialization and data order. Positive "
              "differences favor output training. We report Student 95% intervals across seeds and separate image-group bootstrap "
              "intervals. Correlated patches are not independent replicates. Small pilot variance samples limit power estimates. "
              "Identity, mean replacement and untrained BSF controls test the bottleneck and establish reference outcomes.")
    config = evidence["config"]
    counts = details["split_counts"]
    probe = details["selected_probe"]
    paragraph(f"Actual data: {counts['train']:,} training, {counts['dev']:,} development and {counts['test']:,} test images. "
              f"The selected readout uses C={probe['C']:g}, with development cross-entropy {probe['dev_ce']:.6g} "
              f"and accuracy {probe['dev_accuracy']:.2%}. Actual training exposure is {details['actual_epochs']} epoch(s) per run; "
              f"the effective batch is {config['effective_batch']} complete images, using microbatches of {config['microbatch']}.")
    if evidence["schedule"]:
        schedule = evidence["schedule"]
        paragraph(f"The frozen schedule contains seeds {', '.join(map(str, schedule['seeds']))}, all five conditions, and "
                  f"{schedule['epochs']} common epoch(s). Initial learning rates: " +
                  "; ".join(f"{name}: {lr:g}" for name, lr in schedule["learning_rates"].items()) + ".")
        if schedule["exploratory"]:
            paragraph(f"The pilot calculation requested {schedule['power']['required']} seeds, but only {len(schedule['seeds'])} fit "
                      "the conservative timing forecast. The allocator first stopped, then a disclosed post-pilot amendment implemented "
                      "the planned exploratory fallback before any main run or test evaluation. "
                      f"Pilot-estimated power at five seeds is {schedule['power']['power_at_five']:.1%}, below the 80% target. "
                      "The original deadline was retained; no additional exposure was added when execution was faster than forecast. "
                      "One epoch limits convergence claims. The forecast remains unchanged in the published schedule.")
        paragraph("Training source revision: " + escape(schedule["source_revision"]) + ". Model and implementation pins, "
                  "the complete allocation, split hashes and the original failure record are linked from the repository.", "SmallBody")
    paragraph(f"Measured training across all reported runs totaled {details['total_training_seconds'] / 3600:.3f} hours. "
              f"Peak PyTorch-allocated VRAM was {details['peak_vram_bytes'] / 2**20:.1f} MiB; this excludes other applications and "
              "CUDA allocations outside PyTorch. Preparation, evaluation and reporting are separate from this sum.")
    story.append(PageBreak())
    heading("4. Results and claims")
    paragraph(escape(evidence["narrative"]))
    headers = ["Condition", "Seeds", "KL nats/image", "Accuracy", "NMSE", "Dead blocks"]
    table_data = [[Paragraph(value, styles["SmallBody"]) for value in headers]]
    for row in evidence["summaries"]:
        values = [row["label"], str(row["n_seeds"]), f"{row['kl']:.6g}", f"{row['accuracy']:.2%}",
                  f"{row['nmse']:.5g}", f"{row['dead_fraction']:.2%}"]
        table_data.append([Paragraph(value, styles["SmallBody"]) for value in values])
    table = Table(table_data, colWidths=[135, 45, 83, 64, 64, 70], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf2")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#557287"))]))
    story.append(table)
    story.append(Spacer(1, 12))
    for name, row in evidence["controls"].items():
        paragraph(f"Control {escape(name)}: KL {row['kl']:.6g} nats/image, accuracy {row['accuracy']:.2%}, NMSE {row['nmse']:.6g}.", "SmallBody")
    if "block_kl" in by_condition and "block_mse" in by_condition:
        local, proposed = by_condition["block_mse"], by_condition["block_kl"]
        paragraph(f"Block KL's mean divergence is {100 * (1 - proposed['kl'] / local['kl']):.2f}% lower than block MSE's, "
                  f"while normalized reconstruction error is {100 * (proposed['nmse'] / local['nmse'] - 1):.2f}% higher. "
                  f"Both have {local['parameters']:,} trainable parameters. Accuracy differences are secondary outcomes: "
                  "training matched the teacher distributions and did not optimize held-out class labels. A higher observed "
                  "accuracy than the teacher does not demonstrate broad representation preservation.")
    scalar = evidence["secondary_scalar"]
    if scalar.get("ci95"):
        paragraph(f"Secondary scalar local-minus-KL difference: {scalar['mean_improvement']:.6g} nats/image, "
                  f"with paired seed interval [{scalar['ci95'][0]:.6g}, {scalar['ci95'][1]:.6g}]. "
                  "This is a descriptive secondary comparison without multiplicity correction. "
                  f"Scalar models have {by_condition['scalar_kl']['parameters']:,} parameters, including their additional per-block gains.")
    heading("Paired primary measurements")
    for row in evidence["primary"]["per_seed"]:
        local_ci, proposed_ci = row["local_image_ci95"], row["proposed_image_ci95"]
        paragraph(f"Seed {row['seed']}: local KL {row['local_kl']:.6g}; proposed KL {row['proposed_kl']:.6g}; "
                  f"difference {row['improvement']:+.6g}. " +
                  (f"Image-group 95% intervals: local [{local_ci[0]:.6g}, {local_ci[1]:.6g}], "
                   f"proposed [{proposed_ci[0]:.6g}, {proposed_ci[1]:.6g}]." if local_ci and proposed_ci else ""), "SmallBody")
    paragraph("Image-group intervals use 2,000 percentile bootstrap resamples, conditional on each trained model. "
              "They do not replace the paired training-seed interval and do not pool patches as independent observations.", "SmallBody")
    figure("comparison.png", "Figure 1. Measured held-out output fidelity and patch reconstruction. Seed error bars appear only when more than one seed is available; their absence is not evidence of certainty.")
    figure("paired_effects.png", "Figure 2. All paired primary effects. The mean's 95% seed interval is shown when estimable. Positive values favor KL; zero denotes no improvement.")
    figure("training.png", "Figure 3. All completed runs' training-batch KL against images seen. Panels share a linear scale. Batch fluctuations are retained; no smoothing or favorable epoch selection is used.", width=5.6 * inch)
    figure("maps.png", "Figure 4. Algorithmically selected development feature maps for fixed examples. Maps are normalized separately to unit spatial L2 norm. Feature indices differ across independently fitted dictionaries; no semantic correspondence is assumed.")
    heading("5. Interventions and geometry")
    paragraph("Eight blocks are ranked by development energy. For fixed test examples, remove the strongest selected block, remove a "
              "random active block among the eight closest-energy alternatives, or replace that block's complete patch code with a "
              "predetermined same-class/other-class donor. These are causal interventions on the composite classifier. Changes in KL "
              "measure behavior under these interventions and do not show that the block encodes the class or another semantic factor. "
              "Complete per-example energies, donors and outputs accompany the report. Matching is approximate. A development-selected "
              "block may be inactive on a fixed target while the random alternative is active; these targets are retained and reported.")
    for diagnostic in evidence["diagnostics"]:
        rows = diagnostic["interventions"]
        if rows:
            values = {name: sum(row["kl"][name] - row["kl"]["baseline"] for row in rows) / len(rows)
                      for name in ("remove_selected", "remove_random_matched", "swap_same_class", "swap_other_class")}
            compact = "; ".join(f"{name}: {value:.5g}" for name, value in values.items())
            inactive = sum(row["selected_energy"] == 0 for row in rows)
            paragraph(escape(diagnostic["condition"]) + f", seed {diagnostic['seed']}, {len(rows)} examples ({inactive} inactive selected blocks): " + escape(compact) + " (mean KL change).", "SmallBody")
    heading("6. Discussion, limitations and implications")
    # Local: compare published outcomes; global: distinguish a qualitative analogy from replication.
    # Source: https://arxiv.org/html/2405.12241v2#S3.SS2 (Sections 3.1-3.2 and Table 1).
    paragraph("Comparison with Braun et al. [1]. Their language-model SAEs improve the sparsity/performance tradeoff while "
              "pure end-to-end training reconstructs activations less accurately (Sections 3.1-3.2). In GPT2-small layer 6, "
              "Table 1 reports 69.4 active features and a 0.145 cross-entropy increase for local training, versus 27.5 and "
              "0.144 for end-to-end training: approximately 60% fewer active features at similar performance loss. "
              "These cross-entropy increases are distinct from our forward-KL metric. Their ReLU SAEs use L1 sparsity sweeps; "
              "ours use fixed TopK with tied signed blocks. Their downstream reconstruction penalty (Section 2.4) reduces "
              "later-layer MSE toward the local baseline, whereas our 0.1 MSE penalty acts on insertion-layer patches.")
    if evidence["profile"] == "full" and "block_kl" in by_condition and "block_mse" in by_condition:
        # Local: qualify the current study's comparison; global: preserve its registered seed uncertainty.
        paragraph("Output fidelity and activation reconstruction are distinct outcomes in both studies. " + escape(evidence["narrative"]) +
                  " A positive mean alone is insufficient. Fixed sparsity and a ten-class readout do not establish "
                  "their broader Pareto or interpretability findings. Worse reconstruction alone neither refutes output matching "
                  "nor demonstrates more interpretable features or recovery of the original model's computation.")
    if "block_kl" in by_condition and "block_mse" in by_condition:
        paragraph(f"Block local-MSE versus KL conditional stable rank is {local['stable_rank']:.3f} versus {proposed['stable_rank']:.3f}; "
                  f"normalized-map total variation is {local['map_tv']:.3f} versus {proposed['map_tv']:.3f}, and Dirichlet energy "
                  f"is {local['map_dirichlet']:.3f} versus {proposed['map_dirichlet']:.3f}. These are descriptive geometry diagnostics. "
                  "The readout objective did not preserve the local-MSE solution's reconstruction or spatial smoothness. "
                  "A positive, zero or negative removal/swap effect cannot identify semantic correspondence between independently trained blocks.")
    paragraph("A ten-class softmax constrains at most nine independent logit contrasts per image. Output preservation can therefore "
              "coexist with poor reconstruction, altered spatial maps or discarded information. This study does not test broad DINO "
              "transfer, segmentation, depth or diffusion behavior. The fixed probe and pooling define the measured function. "
              "Max pooling sends gradients only through winning patch coordinates; hard TopK supplies no ordinary gradient to "
              "unselected blocks for a given example. Dead blocks and seed sensitivity are outcomes, not hidden implementation successes.")
    paragraph("Float32 cached final patches make BSF gradients equivalent to online frozen-backbone extraction within measured numerical "
              "tolerance. This does not exercise backpropagation through earlier DINO layers. Dense kernels remain in use, so this work "
              "makes no speedup claim. Pilot power estimates may be optimistic or pessimistic for longer runs. Any budget failure, "
              "missing seed or post-hoc correction must be reported with the saved status and provenance.")
    heading("7. Conclusion")
    paragraph("The implementation supplies a reproducible test of output-KL training for structured sparse vision codes. " + escape(evidence["narrative"]) +
              " Claims are restricted to the measured teacher, data, architecture and compute allocation.")
    heading("References")
    sources = [("1. Braun et al. Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning (2024), v2, Sections 2.4 and 3.1-3.2, Table 1", "https://arxiv.org/html/2405.12241v2"),
               ("2. Fel et al. Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds (2026)", "https://arxiv.org/abs/2606.25234"),
               ("3. Revisiting End To End Sparse Autoencoder Training: A Short Finetune is All You Need (2025)", "https://arxiv.org/abs/2503.17272"),
               ("4. A Deeper Analysis of Block-Sparse Featurizers (2026)", "https://arxiv.org/abs/2608.27515"),
               ("5. DINOv3 reference implementation and model documentation", "https://github.com/facebookresearch/dinov3"),
               ("6. Imagenette dataset source", "https://github.com/fastai/imagenette"),
               ("7. Goodfire implementation, revision 219f121", "https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9"),
               ("8. Apollo implementation, revision 83b4add", "https://github.com/ApolloResearch/e2e_sae/tree/83b4add4d2652a4e5c6775a0e2b752897c871a87"),
               ("9. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders (2025)", "https://proceedings.mlr.press/v267/karvonen25a.html"),
               ("10. Card et al. With Little Power Comes Great Responsibility (2020)", "https://aclanthology.org/2020.emnlp-main.745/"),
               ("11. Lones. How to avoid machine learning pitfalls (2021)", "https://arxiv.org/abs/2108.02497")]
    for label, url in sources:
        paragraph(f'<link href="{url}" color="#1d5c8b">{escape(label)}</link>', "SmallBody")
    heading("Appendix: reproducibility and AI statement")
    paragraph('Code, preregistration, lockfile, run tables and model releases: <link href="https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers" color="#1d5c8b">project repository</link>. '
              "Per-run checkpoints bind preprocessing, the frozen classifier, source revisions and data manifests. Original images and gated backbone weights are not redistributed.")
    paragraph("AI statement: Codex wrote this codebase and paper. Human cleanup and rewriting are in progress. A human provided the exact specification that Codex executed.", "SmallBody")
    document = SimpleDocTemplate(str(destination / "paper.pdf"), pagesize=(612, 792),
                                  rightMargin=50, leftMargin=50, topMargin=48, bottomMargin=44,
                                  title="End-to-End Block-Sparse Featurizers for a Frozen Vision Readout", author="Libor Burian")

    def footer(canvas, document):
        """Local: page numbering; global: make exported evidence easy to navigate."""
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526275"))
        canvas.drawString(50, 25, "Libor Burian | End-to-End Block-Sparse Featurizers")
        canvas.drawRightString(562, 25, str(document.page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
