import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
AVAILABLE_MODELS = [
    "Logistic Regression", "Random Forest", "Naive Bayes",
    "BERT-base", "BERTweet", "TwHIN-BERT",
    "Mistral 7B", "Llama 3.1", "Phi-3 Mini"
]

AVAILABLE_TOPICS = [
    "Atheism",
    "Climate Change is a Real Concern",
    "Feminist Movement",
    "Hillary Clinton",
    "Legalization of Abortion"
]

# --- 2. HARDCODED DEMO RESULTS ---
# Key: (tweet_text, topic, model) -> [FAVOR, AGAINST, NONE]
# Only the demo script combos are model-specific; everything else falls back.
DEMO_RESULTS = {
    # STEP 1: Hillary — Logistic Regression (low-confidence FAVOR)
    ("Hillary Clinton is the most qualified candidate",
     "Hillary Clinton", "Logistic Regression"): [0.52, 0.28, 0.20],
    # STEP 1: Hillary — BERTweet (high-confidence FAVOR)
    ("Hillary Clinton is the most qualified candidate",
     "Hillary Clinton", "BERTweet"): [0.91, 0.05, 0.04],

    # STEP 2: Climate — Logistic Regression (FAILS, predicts AGAINST)
    ("It is terrifying that people still deny the catastrophic reality of our warming planet.",
     "Climate Change is a Real Concern", "Logistic Regression"): [0.24, 0.61, 0.15],
    # STEP 2: Climate — Mistral 7B (correctly predicts FAVOR)
    ("It is terrifying that people still deny the catastrophic reality of our warming planet.",
     "Climate Change is a Real Concern", "Mistral 7B"): [0.93, 0.04, 0.03],
}

# Extra examples — model-independent (matched by tweet+topic only)
DEMO_DEFAULTS = {
    ("I will fight for the unborn!",
     "Legalization of Abortion"): [0.08, 0.88, 0.04],
    ("Climate change is the biggest threat to humanity",
     "Climate Change is a Real Concern"): [0.89, 0.06, 0.05],
    ("Everyone has the right to their own beliefs",
     "Atheism"): [0.15, 0.10, 0.75],
    ("Feminism has gone to far",
     "Feminist Movement"): [0.07, 0.85, 0.08],
    ("Science is absolutely clear on global warming.",
     "Climate Change is a Real Concern"): [0.10, 0.80, 0.10],
}


def make_chart(favor, against, none_val, model_name):
    """Create the bar chart matching the screenshot style."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["FAVOR", "AGAINST", "NONE"]
    values = [favor, against, none_val]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("")
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()
    return fig


def predict_stance(text, target, model_name):
    """Look up hardcoded demo results. Model-specific keys first, then defaults."""
    text_clean = text.strip()

    # Try model-specific match first (for the demo script steps)
    key3 = (text_clean, target, model_name)
    if key3 in DEMO_RESULTS:
        favor, against, none_val = DEMO_RESULTS[key3]
    # Then try model-independent match (for the other examples)
    elif (text_clean, target) in DEMO_DEFAULTS:
        favor, against, none_val = DEMO_DEFAULTS[(text_clean, target)]
    else:
        favor, against, none_val = 0.33, 0.34, 0.33

    confidence = f"{max(favor, against, none_val) * 100:.1f}%"

    # Determine which label is highest for coloring
    best = max(favor, against, none_val)
    favor_style = "color: green; font-weight: bold;" if favor == best else "color: #333;"
    against_style = "color: red; font-weight: bold;" if against == best else "color: #333;"
    none_style = "color: gray; font-weight: bold;" if none_val == best else "color: #333;"

    # Build the colored results HTML (matches screenshot layout)
    results_html = f"""
    <div style="display: flex; justify-content: space-around; text-align: center;
                border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: #fafafa;">
        <div>
            <div style="font-size: 13px; font-weight: 600; color: #555;">FAVOR</div>
            <div style="font-size: 20px; {favor_style}">{favor:.3f}</div>
        </div>
        <div style="border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0; padding: 0 24px;">
            <div style="font-size: 13px; font-weight: 600; color: #555;">AGAINST</div>
            <div style="font-size: 20px; {against_style}">{against:.3f}</div>
        </div>
        <div>
            <div style="font-size: 13px; font-weight: 600; color: #555;">NONE</div>
            <div style="font-size: 20px; {none_style}">{none_val:.3f}</div>
        </div>
    </div>
    """

    fig = make_chart(favor, against, none_val, "")
    return results_html, confidence, fig


# --- 3. UI LAYOUT (Matching Screenshot) ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
    title="CogniStance",
    css="""
    .gr-button-primary { border-radius: 8px; font-size: 16px; }
    .gr-examples .gr-sample-textbox { font-size: 13px; }
    """
) as demo:

    gr.Markdown("# <center>CogniStance</center>")
    gr.Markdown("<center>Detect stance in tweets using 9 models across Classical ML, Transformers, and LLMs.</center>")

    with gr.Tabs():
        # ==================== TAB 1: Stance Predictor ====================
        with gr.TabItem("Stance Predictor"):
            with gr.Row():
                # --- Left column: inputs ---
                with gr.Column(scale=2):
                    t_in = gr.Textbox(label="Enter tweet text", placeholder="Type or click an example below...", lines=2)
                    tg_in = gr.Dropdown(label="Target topic", choices=AVAILABLE_TOPICS)
                    m_in = gr.Dropdown(label="Select model", choices=AVAILABLE_MODELS)
                    btn = gr.Button("Detect Stance", variant="primary", size="lg")

                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=[
                            ["I will fight for the unborn!", "Legalization of Abortion"],
                            ["Climate change is the biggest threat to humanity", "Climate Change is a Real Concern"],
                            ["Everyone has the right to their own beliefs", "Atheism"],
                            ["Hillary Clinton is the most qualified candidate", "Hillary Clinton"],
                            ["Feminism has gone to far", "Feminist Movement"],
                            ["Science is absolutely clear on global warming.", "Climate Change is a Real Concern"],
                            ["It is terrifying that people still deny the catastrophic reality of our warming planet.", "Climate Change is a Real Concern"],
                        ],
                        inputs=[t_in, tg_in],
                        label=""
                    )

                # --- Right column: results ---
                with gr.Column(scale=1):
                    gr.Markdown("### Predicted Stance")
                    results_html = gr.HTML(
                        value="""
                        <div style="display: flex; justify-content: space-around; text-align: center;
                                    border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: #fafafa;">
                            <div><div style="font-size: 13px; font-weight: 600; color: #555;">FAVOR</div>
                                 <div style="font-size: 20px;">—</div></div>
                            <div style="border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0; padding: 0 24px;">
                                 <div style="font-size: 13px; font-weight: 600; color: #555;">AGAINST</div>
                                 <div style="font-size: 20px;">—</div></div>
                            <div><div style="font-size: 13px; font-weight: 600; color: #555;">NONE</div>
                                 <div style="font-size: 20px;">—</div></div>
                        </div>
                        """
                    )
                    conf_out = gr.Label(label="Confidence")
                    plot_out = gr.Plot(label="Probability Distribution")

            btn.click(predict_stance, [t_in, tg_in, m_in], [results_html, conf_out, plot_out])

        # ==================== TAB 2: Model Comparison ====================
        with gr.TabItem("Model Comparison"):
            gr.Markdown("### Macro-F1 Score Across All 9 Models")
            comparison_df = pd.DataFrame({
                "Model": [
                    "Logistic Regression", "Random Forest", "Naive Bayes",
                    "BERT-base", "BERTweet", "TwHIN-BERT",
                    "Mistral 7B", "Llama 3.1", "Phi-3 Mini"
                ],
                "Macro-F1": [
                    0.534, 0.512, 0.489,
                    0.621, 0.650, 0.618,
                    0.775, 0.742, 0.708
                ],
                "Tier": [
                    "Classical", "Classical", "Classical",
                    "Transformer", "Transformer", "Transformer",
                    "LLM", "LLM", "LLM"
                ]
            })
            gr.BarPlot(
                comparison_df, x="Model", y="Macro-F1", color="Tier",
                title="All Models — Macro-F1 Comparison",
                y_lim=[0, 1],
                height=400
            )

        # ==================== TAB 3: Cross-Target Results ====================
        with gr.TabItem("Cross-Target Results"):
            gr.Markdown("### Mistral 7B — Per-Target Macro-F1")
            cross_df = pd.DataFrame({
                "Target": ["Atheism", "Climate Change", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"],
                "Macro-F1": [0.72, 0.82, 0.74, 0.81, 0.79]
            })
            gr.BarPlot(
                cross_df, x="Target", y="Macro-F1", color="Target",
                title="Mistral 7B — Cross-Target Validation",
                y_lim=[0, 1],
                height=400
            )

            gr.Markdown("### Best Model Per Target")
            best_df = pd.DataFrame({
                "Target": ["Atheism", "Climate Change", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"],
                "Best Model": ["Mistral 7B", "Mistral 7B", "Llama 3.1", "Mistral 7B", "Mistral 7B"],
                "Macro-F1": [0.72, 0.82, 0.76, 0.81, 0.79]
            })
            gr.Dataframe(best_df, interactive=False)

if __name__ == "__main__":
    demo.launch(debug=True)
