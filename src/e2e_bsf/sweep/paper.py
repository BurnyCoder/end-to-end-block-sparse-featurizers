"""Global: an explicitly exploratory scientific report; local: measured ReportLab publication.

Source: https://docs.reportlab.com/reportlab/userguide/ch5_platypus/.
"""

from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether


def write_paper(destination, evidence):
    """Render the same evidence as the explorer; never substitute a selected favorable point."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 9
    story = []

    def paragraph(text, style="BodyText"):
        """One typography helper keeps prose and scientific scope consistent."""
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        """Keep headings with the following content through ReportLab's paragraph style."""
        paragraph(text, "Heading2")

    def figure(name, caption, width=500):
        """Preserve scientific aspect ratios and keep captions attached to their figures."""
        from PIL import Image as PILImage
        path = destination / "figures" / name
        if path.exists():
            with PILImage.open(path) as opened:
                height = width * opened.height / opened.width
            story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["Small"])]))

    paragraph("Downstream Reconstruction and Sparsity in End-to-End Vision Featurizers", "Title")
    paragraph("Libor Burian", "Heading2")
    paragraph(escape(evidence["label"]))
    heading("Abstract")
    paragraph("We compare local reconstruction, output KL, and output KL plus raw downstream reconstruction for Grassmannian "
              "block-sparse featurizers inserted after DINOv3 block 6. All 201 tokens are reconstructed before the remaining "
              "frozen transformer blocks and a fixed patch-only Imagenette classifier. Five active-block budgets and three "
              "downstream weights define 25 operating points with one shared initialization seed. " + escape(evidence["narrative"]))
    heading("1. Context, motivation and related work")
    paragraph("Braun et al. [1] optimize output matching and optionally reconstruct downstream layers to distinguish output "
              "fidelity from activation fidelity. Their language-model sparsity sweeps motivate this vision adaptation. "
              "We retain Goodfire's released tied, signed Grassmannian charts [2,3], rather than conventional ReLU SAEs. "
              "The earlier final-layer study [4] is historical evidence, with a different insertion site and token scope. "
              "This follow-up does not retrospectively change that experiment.")
    heading("2. Model, data and objective")
    paragraph("The backbone is DINOv3 ViT-B/16 [5], revision 5931719e67bbdb9737e363e781fb0c67687896bc, using its pinned "
              "224-pixel processor. The original model is this backbone plus the previously fitted, frozen ten-class linear "
              "readout of maximum absolute final patch coordinates. CLS and four register tokens are excluded from the readout. "
              "At the insertion site, however, the BSF replaces every token, preventing a clean-special-token bypass.")
    counts = evidence["split_counts"]
    paragraph(f"We reuse the original duplicate-screened Imagenette partitions: {counts['train']:,} training, {counts['dev']:,} development, "
              f"and {counts['test']:,} test images. The test set was analyzed previously and is explicitly reused; it is not a fresh confirmatory sample. "
              "New positional means and one RMS scale are fitted on training-only block-6 states. The complete state is normalized "
              "before the BSF and restored to raw units before entering the frozen suffix. No reconstruction error is added back.")
    paragraph("Each dictionary contains 4,096 blocks of width 3 (9,441,280 parameters). The retained block counts are "
              "k={4,8,16,32,64}, corresponding to {12,24,48,96,192} active coordinates per token. All runs start from the same "
              "random seed 200 and use identical image order. Differentiable QR, tied signed encoder/decoder weights, hard block "
              "TopK and one unconstrained learned gain per block follow the released implementation [3].")
    paragraph("Local training minimizes normalized insertion MSE. Output training minimizes KL(p_teacher || q_inserted), "
              "averaged over images at temperature 1. The downstream conditions use L=KL+(beta/6) sum(j=7..12) MSE(Hhat_j,H_j), "
              "where beta={0.0001,0.001,0.01}. Each MSE averages all images, 201 tokens and 768 coordinates in the raw residual "
              "state after the indicated block, before terminal normalization. Targets are detached; only BSF parameters train. "
              "These conditions contain no insertion-MSE term. Training-variance-normalized downstream errors are diagnostics only.")
    heading("3. Allocation, optimization and analysis")
    schedule = evidence["schedule"]
    paragraph("The new immutable wall-time limit is 24 hours: two for preparation/verification/pilots, training until hour 22, "
              "and two for final evaluation/reporting. Each cell receives the same pilot LR grid {0.0001,0.0003} with seed 900 "
              "on fixed class-balanced training/development subsets. Development KL selects LR, with a lower-LR tie break. "
              "The complete grid and common exposure are frozen using measured throughput plus 25% allowance, without consulting test outcomes.")
    if schedule:
        paragraph(f"Frozen exposure: {schedule['epochs']} epoch(s) for each of 25 main runs; effective batch 32, microbatch "
                  f"{schedule['microbatch']}. Adam uses gradient clipping at 1 and cosine LR decay to 10% of its initial value. "
                  "Runs advance one epoch per cell per round and use the final scheduled checkpoint. No run is added to obtain significance.")
        paragraph("Selected learning rates by cell: " + escape("; ".join(f"{key}: {value:g}" for key, value in schedule["learning_rates"].items())), "Small")
    paragraph("The primary endpoint is local-minus-pure-KL normalized area under KL versus log2 active coordinates, with "
              "trapezoidal integration only over the five measured points. Positive differences favor KL. We use 2,000 shared "
              "paired image-group bootstrap draws for conditional 95% intervals. There is one training seed, so no seed-level "
              "uncertainty or power claim is available. Reconstruction-weight comparisons are descriptive secondary analyses. "
              "A mean advantage does not imply dominance at every sparsity or at unmeasured operating points.")
    heading("4. Results, controls and claims")
    paragraph(escape(evidence["narrative"]))
    headers = ["Objective", "k", "KL", "CE change", "Accuracy", "Local NMSE", "Suffix MSE"]
    data = [[Paragraph(v, styles["Small"]) for v in headers]]
    for row in evidence["runs"]:
        values = [row["label"], str(row["k"]), f"{row['kl']:.5g}", f"{row['ce_delta']:.4g}",
                  f"{row['accuracy']:.2%}", f"{row['insertion_nmse']:.4g}", f"{row['downstream_mse']:.5g}"]
        data.append([Paragraph(v, styles["Small"]) for v in values])
    table = Table(data, colWidths=[131, 25, 62, 62, 56, 65, 65], repeatRows=1)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf2")),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(table)
    story.append(Spacer(1, 10))
    for control in evidence["controls"]:
        paragraph(f"Control {escape(control['control'])}: KL {control['kl']:.6g}; accuracy {control['accuracy']:.2%}; "
                  f"insertion NMSE {control['insertion_nmse']:.5g}; downstream raw MSE {control['downstream_mse']:.6g}.", "Small")
    for label, effect in evidence["secondary"].items():
        paragraph(f"Secondary {escape(label)}: local-minus-proposed normalized AUC {effect['mean_improvement']:.6g}, "
                  f"conditional interval [{effect['ci95'][0]:.6g}, {effect['ci95'][1]:.6g}].", "Small")
    for label, thresholds in evidence["thresholds"].items():
        paragraph(escape(label) + ": smallest measured active-coordinate budget at KL thresholds " +
                  escape("; ".join(f"{key}: {value if value is not None else 'unattained'}" for key, value in thresholds.items())) +
                  ". No interpolation or extrapolation.", "Small")
    paragraph("Descriptive sparsity/KL Pareto frontier (minimize both; retain equal points): " +
              escape(", ".join(evidence["pareto"].get("active_coordinates", []))) +
              ". This observed frontier does not adjust for estimation uncertainty.", "Small")
    figure("curves.png", "Figure 1. All measured operating points. Bars show conditional image-group intervals; they do not measure seed variation.")
    figure("layers.png", "Figure 2. Reconstruction across layers at k=16; raw and training-variance-normalized errors answer different questions.")
    figure("training.png", "Figure 3. Every main run's unsmoothed training KL, on common linear axes. No favorable checkpoint selection.", width=420)
    heading("5. Interventions, discussion and limitations")
    figure("maps.png", "Figure 4. Fixed development examples at k=16. Selected blocks differ between dictionaries; maps are separately normalized and imply no semantic correspondence.")
    paragraph("At k=16, blocks are selected by development-patch energy and displayed on fixed development examples. "
              "Removal and predetermined same-class/other-class donor swaps alter the selected block over all 201 positions. "
              "The random control samples among eight nearest-energy active alternatives. Matching is approximate, and inactive "
              "selected blocks remain in the results. These are interventions on this composite classifier, not evidence of semantic correspondence.")
    for diagnostic in evidence["diagnostics"]:
        rows = diagnostic["interventions"]
        if rows:
            values = {key: sum(row["kl"][key] - row["kl"]["baseline"] for row in rows) / len(rows)
                      for key in ("remove_selected", "remove_random_matched", "swap_same_class", "swap_other_class")}
            paragraph(escape(diagnostic["label"]) + f", {len(rows)} targets, mean KL changes: " +
                      escape("; ".join(f"{key}={value:.5g}" for key, value in values.items())), "Small")
    paragraph("A ten-class readout constrains only a narrow portion of DINO behavior. Lower KL, lower reconstruction error, "
              "or sparse codes do not establish full representation preservation, genuine concepts, or faster inference. "
              "Raw downstream MSE can emphasize high-variance layers and special tokens; separate patch/special and normalized "
              "diagnostics expose this issue. One seed, short training, reused test data, and a two-value LR search limit generalization. "
              "Objective comparisons include their selected LRs. Known study duplicates are isolated; unknown backbone pretraining overlap is unresolved.")
    heading("6. Conclusion")
    paragraph(escape(evidence["narrative"]) + " The complete curves, including unfavorable points and unattained thresholds, define the scope of the evidence.")
    heading("References and reproducibility")
    for label, url in [("1. Braun et al. End-to-End Sparse Dictionary Learning (2024), Sections 2.4 and 3", "https://arxiv.org/html/2405.12241v2"),
                       ("2. Fel et al. Block-Sparse Featurizers (2026)", "https://arxiv.org/abs/2606.25234"),
                       ("3. Goodfire implementation, revision 219f121", "https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9"),
                       ("4. Project, original study, full artifacts and preregistration", "https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers"),
                       ("5. DINOv3 implementation", "https://github.com/facebookresearch/dinov3"),
                       ("6. Imagenette dataset", "https://github.com/fastai/imagenette")]:
        paragraph(f'<link href="{url}" color="#1d5c8b">{escape(label)}</link>', "Small")
    paragraph("Safe model bundles include BSF weights, the fixed classifier, normalization, full predictions, histories and provenance. "
              "Original photos, raw activation caches and gated backbone weights are not redistributed. Local checkpoints additionally preserve optimizer and RNG state.")
    heading("AI statement")
    paragraph("Codex wrote this codebase and paper. Human cleanup and rewriting are in progress. A human provided the exact specification that Codex executed.", "Small")

    def footer(canvas, document):
        """Stable page labels make exported results referenceable."""
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 25, "Libor Burian | Downstream reconstruction and sparsity")
        canvas.drawRightString(562, 25, str(document.page))

    document = SimpleDocTemplate(str(destination / "paper.pdf"), pagesize=(612, 792), leftMargin=48, rightMargin=48,
                                  topMargin=45, bottomMargin=44, author="Libor Burian",
                                  title="Downstream Reconstruction and Sparsity in End-to-End Vision Featurizers")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
