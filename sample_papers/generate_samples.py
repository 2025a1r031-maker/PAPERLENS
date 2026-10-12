"""
sample_papers/generate_samples.py
Generates realistic academic research paper PDFs for testing and one-click demonstration
using PyMuPDF. Creates rich multi-page documents with title, abstract, methodology,
experimental results, and conclusions.
"""

import os
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def create_paper_1(output_path: str):
    doc = fitz.open()

    # --- Page 1 ---
    page1 = doc.new_page(width=595, height=842)  # Standard A4
    y = 60

    # Header / Conference Tag
    page1.insert_text(fitz.Point(50, y), "PROCEEDINGS OF THE INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML)", fontsize=8, color=(0.4, 0.4, 0.4))
    y += 40

    # Title
    page1.insert_text(fitz.Point(50, y), "Attention-Based Deep Neural Architectures for", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 24
    page1.insert_text(fitz.Point(50, y), "Sequence Transduction and Semantic Representation", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 30

    # Authors
    page1.insert_text(fitz.Point(50, y), "Ashish Vaswani, Elena Chen, Rajesh Mercer", fontsize=11, color=(0.2, 0.2, 0.2))
    y += 16
    page1.insert_text(fitz.Point(50, y), "Department of Computer Science, Institute for Advanced Artificial Intelligence", fontsize=9, color=(0.4, 0.4, 0.4))
    y += 35

    # Abstract Box
    page1.draw_rect(fitz.Rect(45, y - 10, 550, y + 115), color=(0.85, 0.88, 0.95), fill=(0.96, 0.97, 1.0))
    page1.insert_text(fitz.Point(55, y + 8), "ABSTRACT", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.4))

    abstract_text = (
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
        "that include an encoder and a decoder. We propose a simple network architecture, the Transformer, based solely "
        "on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation "
        "tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time "
        "to train. The primary problem addressed is the sequential computation bottleneck inherent in recurrent architectures, "
        "which prevents parallelization across long training sequences. Our methodology utilizes multi-head self-attention and "
        "positional encodings to capture global dependencies without distance constraints. The principal contribution is establishing "
        "that attention alone achieves state-of-the-art translation accuracy while reducing training compute by an order of magnitude."
    )
    rect_abs = fitz.Rect(55, y + 20, 540, y + 105)
    page1.insert_textbox(rect_abs, abstract_text, fontsize=9.5, fontname="times-roman", lineheight=1.3)
    y += 135

    # Section 1: Introduction
    page1.insert_text(fitz.Point(50, y), "1. INTRODUCTION", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 18

    intro_text = (
        "Recurrent neural networks, long short-term memory (LSTM), and gated recurrent neural networks have been firmly "
        "established as state-of-the-art approaches in sequence modeling. However, sequential computation remains an inherent "
        "bottleneck in recurrent language models, preventing parallelization within training examples. This fundamental limitation "
        "becomes critical at longer sequence lengths, as memory constraints limit batching across examples.\n\n"
        "Recent work has achieved remarkable improvements in computational efficiency through factorization tricks and conditional "
        "computation, but the fundamental constraint of sequential execution remains. Attention mechanisms have become an integral part "
        "of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in input sequences.\n\n"
        "In this work, we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention "
        "mechanism to draw global dependencies between input and output representations. The Transformer allows for significantly more "
        "parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours."
    )
    rect_intro = fitz.Rect(50, y, 545, 800)
    page1.insert_textbox(rect_intro, intro_text, fontsize=10, fontname="times-roman", lineheight=1.35)

    # --- Page 2 ---
    page2 = doc.new_page(width=595, height=842)
    y2 = 60

    page2.insert_text(fitz.Point(50, y2), "2. MODEL ARCHITECTURE AND METHODOLOGY", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y2 += 18

    method_text = (
        "Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input "
        "sequence of symbol representations to a sequence of continuous representations. Given this continuous representation, the "
        "decoder then generates an output sequence of symbols one element at a time.\n\n"
        "Our proposed architecture follows this overall framework using stacked self-attention and point-wise, fully connected layers "
        "for both the encoder and decoder. The encoder is composed of a stack of N identical layers. Each layer has two sub-layers: "
        "a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network.\n\n"
        "An attention function can be described as mapping a query and a set of key-value pairs to an output. The output is computed "
        "as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query "
        "with the corresponding key. Specifically, we compute scaled dot-product attention over query matrix Q, key matrix K, and value matrix V.\n\n"
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. "
        "With a single attention head, averaging inhibits this capacity. In addition, we apply residual connections around each of the sub-layers, "
        "followed by layer normalization to ensure gradient stability throughout training."
    )
    rect_meth = fitz.Rect(50, y2, 545, y2 + 280)
    page2.insert_textbox(rect_meth, method_text, fontsize=10, fontname="times-roman", lineheight=1.35)
    y2 += 300

    # Section 3: Experimental Results
    page2.insert_text(fitz.Point(50, y2), "3. EXPERIMENTAL EVALUATION AND RESULTS", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y2 += 18

    results_text = (
        "On the WMT 2014 English-to-German translation task, the big transformer model out-performs the best previously reported models "
        "(including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French "
        "translation task, our big model achieves a BLEU score of 41.0, outperforming all previously published single models at 1/4 the training cost.\n\n"
        "Training was carried out on standard NVIDIA P100 GPUs using the Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 10^-9. "
        "We varied the learning rate over the course of training according to a warm-up schedule. Model checkpoints were averaged over the last "
        "five evaluations to produce the final inference parameters."
    )
    rect_res = fitz.Rect(50, y2, 545, 800)
    page2.insert_textbox(rect_res, results_text, fontsize=10, fontname="times-roman", lineheight=1.35)

    # --- Page 3 ---
    page3 = doc.new_page(width=595, height=842)
    y3 = 60

    page3.insert_text(fitz.Point(50, y3), "4. CONCLUSION AND FUTURE WORK", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y3 += 18

    conclusion_text = (
        "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing recurrent "
        "layers most commonly used in encoder-decoder architectures with multi-head self-attention. For translation tasks, the Transformer "
        "can be trained significantly faster than architectures based on recurrent or convolutional layers.\n\n"
        "The primary contribution of this research is establishing that attention alone is sufficient for sequence modeling, achieving superior "
        "translation accuracy while eliminating sequential execution bottlenecks. We plan to extend the Transformer to problems involving input "
        "and output modalities other than text and investigate local, restricted attention mechanisms to efficiently handle massive inputs such as images and audio."
    )
    rect_conc = fitz.Rect(50, y3, 545, y3 + 150)
    page3.insert_textbox(rect_conc, conclusion_text, fontsize=10, fontname="times-roman", lineheight=1.35)
    y3 += 170

    # References
    page3.insert_text(fitz.Point(50, y3), "5. REFERENCES", fontsize=11, fontname="helv", color=(0.3, 0.3, 0.3))
    y3 += 16
    refs = (
        "[1] Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR.\n"
        "[2] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.\n"
        "[3] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. NeurIPS.\n"
        "[4] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR."
    )
    rect_ref = fitz.Rect(50, y3, 545, 800)
    page3.insert_textbox(rect_ref, refs, fontsize=8.5, fontname="times-roman", lineheight=1.4)

    # Metadata
    doc.set_metadata({
        "title": "Attention-Based Deep Neural Architectures for Sequence Transduction",
        "author": "Ashish Vaswani, Elena Chen, Rajesh Mercer",
        "subject": "Deep Learning and Natural Language Processing",
        "keywords": "Transformer, Attention Mechanism, Neural Machine Translation, Deep Learning",
    })

    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


def create_paper_2(output_path: str):
    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    y = 60

    page1.insert_text(fitz.Point(50, y), "IEEE TRANSACTIONS ON QUANTUM ENGINEERING", fontsize=8, color=(0.4, 0.4, 0.4))
    y += 40

    page1.insert_text(fitz.Point(50, y), "Fault-Tolerant Quantum Error Mitigation in", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 24
    page1.insert_text(fitz.Point(50, y), "Surface Code Architectures and NISQ Devices", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 30

    page1.insert_text(fitz.Point(50, y), "S. Thorne, M. Al-Farabi, H. Nakamura", fontsize=11, color=(0.2, 0.2, 0.2))
    y += 16
    page1.insert_text(fitz.Point(50, y), "Center for Quantum Computing & Department of Applied Physics", fontsize=9, color=(0.4, 0.4, 0.4))
    y += 35

    page1.draw_rect(fitz.Rect(45, y - 10, 550, y + 115), color=(0.85, 0.88, 0.95), fill=(0.96, 0.97, 1.0))
    page1.insert_text(fitz.Point(55, y + 8), "ABSTRACT", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.4))

    abstract_text = (
        "Quantum computing systems are fundamentally vulnerable to environmental decoherence and quantum gate noise. "
        "This paper addresses the central problem of preserving quantum coherence during deep circuit execution on noisy "
        "intermediate-scale quantum (NISQ) hardware. We introduce a hybrid quantum error mitigation framework combining "
        "surface code stabilizer measurement cycles with zero-noise extrapolation techniques. Our methodology incorporates "
        "probabilistic error cancellation circuits that reconstruct ideal expectation values from intentionally amplified noise profiles. "
        "The primary contribution is demonstrating an order-of-magnitude reduction in logical qubit error rates below the fault-tolerant "
        "threshold theorem boundary, enabling reliable variational quantum eigensolver execution on 64-qubit processors."
    )
    rect_abs = fitz.Rect(55, y + 20, 540, y + 105)
    page1.insert_textbox(rect_abs, abstract_text, fontsize=9.5, fontname="times-roman", lineheight=1.3)
    y += 135

    page1.insert_text(fitz.Point(50, y), "1. INTRODUCTION AND QUANTUM DECOHERENCE", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y += 18

    intro_text = (
        "Physical qubits fabricated using superconducting transmon loops or trapped ions suffer from continuous environmental "
        "phase damping and amplitude relaxation. Quantum error correction codes, particularly topological surface codes, offer "
        "a theoretical pathway toward fault-tolerant computing by encoding a single logical qubit across multiple physical qubits.\n\n"
        "However, physical error rates in contemporary NISQ hardware frequently exceed the fault-tolerance threshold of approximately 1%. "
        "Consequently, active syndrome extraction can inadvertently introduce more noise than it suppresses. Quantum error mitigation (QEM) "
        "has emerged as an essential alternative to bridge the gap between noisy physical processors and fully fault-tolerant architectures.\n\n"
        "In this study, we present a unified framework that couples stabilizer syndrome measurements with zero-noise extrapolation to "
        "systematically mitigate both coherent control errors and stochastic Pauli channel noise."
    )
    rect_intro = fitz.Rect(50, y, 545, 800)
    page1.insert_textbox(rect_intro, intro_text, fontsize=10, fontname="times-roman", lineheight=1.35)

    # --- Page 2 ---
    page2 = doc.new_page(width=595, height=842)
    y2 = 60

    page2.insert_text(fitz.Point(50, y2), "2. ERROR MITIGATION METHODOLOGY", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y2 += 18

    method_text = (
        "Our methodology relies on intentional pulse stretching and Clifford twirling to map arbitrary physical noise channels into "
        "symmetric Pauli error channels. For each quantum circuit layer, unitary operations are conjugated with randomly selected Pauli operators.\n\n"
        "The zero-noise extrapolation protocol executes target quantum circuits at variable noise amplification factors lambda in {1, 1.5, 2.0, 2.5}. "
        "Polynomial and Richardson extrapolation techniques are then applied to fit the expectation value trajectory and estimate the zero-noise limit.\n\n"
        "Simultaneously, ancillary syndrome qubits monitor discrete bit-flip (X) and phase-flip (Z) errors using minimum-weight perfect matching "
        "(MWPM) decoding algorithms executed on an FPGA controller with sub-microsecond latency."
    )
    rect_meth = fitz.Rect(50, y2, 545, y2 + 250)
    page2.insert_textbox(rect_meth, method_text, fontsize=10, fontname="times-roman", lineheight=1.35)
    y2 += 270

    page2.insert_text(fitz.Point(50, y2), "3. EXPERIMENTAL BENCHMARKING AND FINDINGS", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y2 += 18

    results_text = (
        "We benchmarked our hybrid protocol on a 64-qubit superconducting quantum processor performing ground-state energy estimation "
        "for molecular lithium hydride (LiH). Unmitigated circuits exhibited an energy error of 45.2 mHa due to two-qubit cross-resonance gate infidelity.\n\n"
        "Applying our stabilizer mitigation framework reduced the energy error to 1.4 mHa, successfully entering the chemical accuracy regime (< 1.6 mHa). "
        "Furthermore, logical qubit state preservation showed a 12x lifetime improvement compared to bare physical transmon qubits under identical cryostat temperatures."
    )
    rect_res = fitz.Rect(50, y2, 545, 800)
    page2.insert_textbox(rect_res, results_text, fontsize=10, fontname="times-roman", lineheight=1.35)

    # --- Page 3 ---
    page3 = doc.new_page(width=595, height=842)
    y3 = 60

    page3.insert_text(fitz.Point(50, y3), "4. CONCLUSION AND PRACTICAL IMPLICATIONS", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.2))
    y3 += 18

    conclusion_text = (
        "We have demonstrated that combining topological surface code stabilizer cycles with zero-noise extrapolation enables quantum "
        "algorithms to achieve computational utility on current noisy hardware. The primary contribution of this work is demonstrating "
        "that chemical accuracy is achievable without requiring millions of physical qubits.\n\n"
        "This result provides practical validation for near-term variational quantum algorithms and establishes a viable migration pathway "
        "toward large-scale fault-tolerant quantum computation."
    )
    rect_conc = fitz.Rect(50, y3, 545, y3 + 140)
    page3.insert_textbox(rect_conc, conclusion_text, fontsize=10, fontname="times-roman", lineheight=1.35)
    y3 += 160

    page3.insert_text(fitz.Point(50, y3), "5. REFERENCES", fontsize=11, fontname="helv", color=(0.3, 0.3, 0.3))
    y3 += 16
    refs = (
        "[1] Fowler, A. G., et al. (2012). Surface codes: Towards practical large-scale quantum computation. Phys. Rev. A.\n"
        "[2] Temme, K., Bravyi, S., & Gambetta, J. M. (2017). Error mitigation for short-depth quantum circuits. Phys. Rev. Lett.\n"
        "[3] Kandala, A., et al. (2019). Error mitigation extends the computational reach of a quantum processor. Nature.\n"
        "[4] Preskill, J. (2018). Quantum Computing in the NISQ era and beyond. Quantum, 2, 79."
    )
    rect_ref = fitz.Rect(50, y3, 545, 800)
    page3.insert_textbox(rect_ref, refs, fontsize=8.5, fontname="times-roman", lineheight=1.4)

    doc.set_metadata({
        "title": "Fault-Tolerant Quantum Error Mitigation in Surface Code Architectures",
        "author": "S. Thorne, M. Al-Farabi, H. Nakamura",
        "subject": "Quantum Computing and Applied Physics",
        "keywords": "Quantum Error Correction, Surface Codes, NISQ Devices, Zero-Noise Extrapolation",
    })

    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    create_paper_1(os.path.join(out_dir, "attention_mechanism_transformer.pdf"))
    create_paper_2(os.path.join(out_dir, "quantum_error_correction_algorithms.pdf"))
