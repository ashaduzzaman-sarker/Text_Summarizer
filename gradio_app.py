"""Enhanced Gradio web interface for text summarization."""

import gradio as gr
from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.logging.logger import logger
from pathlib import Path
import time


# Initialize predictor with error handling
predictor = None
initialization_error = None

try:
    model_path = "artifacts/model_trainer/final_model"
    
    if not Path(model_path).exists():
        initialization_error = f"❌ Model not found at {model_path}. Please train the model first."
        logger.error(initialization_error)
    else:
        predictor = PredictionPipeline(model_path=model_path)
        logger.info(f"✅ Prediction pipeline initialized on {predictor.device}")
        
except Exception as e:
    initialization_error = f"❌ Failed to load model: {str(e)}"
    logger.error(initialization_error)


def summarize_text(
    text: str,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 2.0
) -> tuple:
    """Summarize text and return summary with statistics.
    
    Args:
        text: Input article text
        max_length: Maximum summary length
        min_length: Minimum summary length
        num_beams: Beam search width
        length_penalty: Length penalty for generation
        
    Returns:
        Tuple of (summary, statistics, processing_time)
    """
    # Check for initialization errors
    if initialization_error:
        return initialization_error, "", ""
    
    # Validate input
    if not text or len(text.strip()) < 50:
        return "⚠️ Please enter at least 50 characters of text.", "", ""
    
    if predictor is None:
        return "❌ Model not loaded. Please check logs.", "", ""
    
    # Validate parameters
    if min_length >= max_length:
        return "⚠️ Min length must be less than max length.", "", ""
    
    try:
        start_time = time.time()
        
        # Generate summary
        summary = predictor.predict(
            text=text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            use_cache=True
        )
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        original_length = len(text)
        summary_length = len(summary)
        compression_ratio = round(summary_length / original_length * 100, 2)
        word_count_original = len(text.split())
        word_count_summary = len(summary.split())
        
        stats = f"""
### 📊 Statistics

| Metric | Value |
|--------|-------|
| **Original Length** | {original_length:,} characters |
| **Summary Length** | {summary_length:,} characters |
| **Compression Ratio** | {compression_ratio}% |
| **Original Words** | {word_count_original:,} words |
| **Summary Words** | {word_count_summary:,} words |
| **Word Reduction** | {word_count_original - word_count_summary:,} words ({round((1 - word_count_summary/word_count_original)*100, 1)}%) |
        """
        
        time_info = f"""
### ⏱️ Performance
- **Processing Time**: {processing_time:.2f} seconds
- **Device**: {predictor.device}
        """
        
        return summary, stats, time_info
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"❌ Error: {str(e)}", "", ""


# Sample articles for demo
EXAMPLES = [
    [
        """(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze medal. Bolt has now collected eight gold medals at world championships, having previously won the 100m and 200m races in the Russian capital. "I'm proud of myself and I'll continue to work to dominate for as long as possible," Bolt said, having previously expressed his intention to carry on until the 2016 Rio Olympics. Victory was never in doubt once he got the baton safely in hand from Ashmeade, while Gatlin and the United States third leg runner Rakieem Salaam failed to get the baton away cleanly. Earlier, Jamaica's women underlined their dominance in the sprint events by winning the 4x100m relay gold, anchored by Shelly-Ann Fraser-Pryce, who like Bolt was completing a triple.""",
        128, 30, 4, 2.0
    ],
    [
        """Scientists at MIT have developed a groundbreaking new battery technology that could revolutionize electric vehicles. The lithium-metal battery uses a solid electrolyte instead of the traditional liquid version, making it safer and more energy-dense. Researchers claim the new batteries can store up to twice as much energy as current lithium-ion batteries while charging in half the time. The technology could extend electric vehicle range to over 500 miles on a single charge and reduce charging times to under 15 minutes. The team expects commercial production to begin within three years, potentially transforming the automotive industry and accelerating the transition to sustainable transportation.""",
        128, 30, 4, 2.0
    ],
    [
        """The World Health Organization announced today that global COVID-19 cases have declined by 40% over the past month, marking a significant milestone in the pandemic response. Health officials attribute the decrease to widespread vaccination efforts, improved treatments, and continued public health measures. More than 5 billion vaccine doses have been administered worldwide, with vaccination rates exceeding 70% in many developed nations. However, WHO experts warn that new variants continue to emerge and emphasize the importance of maintaining vigilance. The organization is calling for increased vaccine equity to ensure developing nations have access to immunization programs.""",
        128, 30, 4, 2.0
    ]
]


# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

#component-0 {
    max-width: 1200px;
    margin: 0 auto;
}

.header-text {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}

.stats-box {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}
"""


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Text Summarizer") as demo:
    
    # Header
    gr.HTML("""
        <div class="header-text">
            <h1 style="margin:0; font-size: 2.5em;">📝 AI Text Summarizer</h1>
            <p style="margin:10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Powered by Fine-tuned BART Model on CNN/DailyMail Dataset
            </p>
        </div>
    """)
    
    # Show initialization status
    if initialization_error:
        gr.Warning(initialization_error)
    else:
        gr.Info(f"✅ Model loaded successfully on {predictor.device}")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📄 Input Text")
            text_input = gr.Textbox(
                label="Article/Text to Summarize",
                placeholder="Paste your article here (minimum 50 characters)...\n\nTip: Works best with news articles, reports, and formal text.",
                lines=12,
                max_lines=20
            )
            
            gr.Markdown("### ⚙️ Generation Parameters")
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=30,
                    maximum=512,
                    value=128,
                    step=10,
                    label="Max Length",
                    info="Maximum summary length in tokens"
                )
                
                min_length_slider = gr.Slider(
                    minimum=10,
                    maximum=256,
                    value=30,
                    step=5,
                    label="Min Length",
                    info="Minimum summary length in tokens"
                )
            
            with gr.Row():
                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Beam Search Width",
                    info="Higher = better quality, slower (4 recommended)"
                )
                
                length_penalty_slider = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="Length Penalty",
                    info="Encourages longer summaries (2.0 recommended)"
                )
            
            with gr.Row():
                summarize_btn = gr.Button(
                    "✨ Generate Summary",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                clear_btn = gr.ClearButton(
                    [text_input, max_length_slider, min_length_slider, 
                     num_beams_slider, length_penalty_slider],
                    value="🗑️ Clear All",
                    size="lg",
                    scale=1
                )
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### ✨ Generated Summary")
            summary_output = gr.Textbox(
                label="Summary",
                lines=12,
                max_lines=20,
                show_copy_button=True
            )
            
            stats_output = gr.Markdown(
                label="Statistics",
                value=""
            )
            
            time_output = gr.Markdown(
                label="Performance",
                value=""
            )
    
    # Examples section
    gr.Markdown("---")
    gr.Markdown("### 📚 Try These Example Articles")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, max_length_slider, min_length_slider, 
                num_beams_slider, length_penalty_slider],
        outputs=[summary_output, stats_output, time_output],
        fn=summarize_text,
        cache_examples=False,
        label="Click an example to load it"
    )
    
    # Event handlers
    summarize_btn.click(
        fn=summarize_text,
        inputs=[text_input, max_length_slider, min_length_slider, 
                num_beams_slider, length_penalty_slider],
        outputs=[summary_output, stats_output, time_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #6c757d;">
        <p><strong>Model Information</strong></p>
        <p>Architecture: BART-large-CNN (406M parameters) | Dataset: CNN/DailyMail</p>
        <p>Performance: ROUGE-1: 0.42 | ROUGE-2: 0.19 | ROUGE-L: 0.30</p>
    </div>
    """)


# Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True, #False,  # Set True to create public link
        show_error=True,
        show_api=True
    )