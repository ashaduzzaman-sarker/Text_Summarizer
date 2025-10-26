# ============================================================================
# gradio_app.py (Gradio Interface)
# ============================================================================
"""Gradio interface for text summarization."""

import gradio as gr
from src.textSummarizer.components.prediction import PredictionPipeline
from src.textSummarizer.logging.logger import logger

# Initialize prediction pipeline
logger.info("Loading model for Gradio interface...")
predictor = PredictionPipeline()
logger.info("Model loaded successfully")


def summarize_text(
    text: str,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4
) -> str:
    """Generate summary using prediction pipeline.
    
    Args:
        text: Input article text
        max_length: Maximum summary length
        min_length: Minimum summary length
        num_beams: Beam search width
        
    Returns:
        Generated summary
    """
    try:
        if not text or len(text.strip()) < 50:
            return "Please provide text with at least 50 characters."
        
        summary = predictor.predict(
            text=text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"Error: {str(e)}"


# Sample article for demo
SAMPLE_ARTICLE = """
(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze after Britain were disqualified for a faulty handover. The 26-year-old Bolt has now collected eight gold medals at world championships, equaling the record held by American trio Carl Lewis, Michael Johnson and Allyson Felix, not to mention the small matter of six Olympic titles. The relay triumph followed individual successes in the 100 and 200 meters in the Russian capital. "I'm proud of myself and I'll continue to work to dominate for as long as possible," Bolt said, having previously expressed his intention to carry on until the 2016 Rio Olympics. Victory was never seriously in doubt once he got the baton safely in hand from Ashmeade, while Gatlin and the United States third leg runner Rakieem Salaam had problems.
"""


# Create Gradio interface
with gr.Blocks(title="Text Summarizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📝 Text Summarizer")
    gr.Markdown("Generate concise summaries using fine-tuned BART model")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Article",
                placeholder="Paste your article text here (minimum 50 characters)...",
                lines=15,
                value=SAMPLE_ARTICLE
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=30,
                    maximum=256,
                    value=128,
                    step=1,
                    label="Maximum Summary Length"
                )
                min_length = gr.Slider(
                    minimum=10,
                    maximum=128,
                    value=30,
                    step=1,
                    label="Minimum Summary Length"
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Beam Search Width (higher = better quality, slower)"
                )
            
            summarize_btn = gr.Button("Generate Summary", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Summary",
                lines=15,
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            [SAMPLE_ARTICLE, 128, 30, 4],
        ],
        inputs=[input_text, max_length, min_length, num_beams],
        outputs=output_text,
        fn=summarize_text,
        cache_examples=False
    )
    
    # Connect button to function
    summarize_btn.click(
        fn=summarize_text,
        inputs=[input_text, max_length, min_length, num_beams],
        outputs=output_text
    )
    
    gr.Markdown("""
    ### About
    This summarizer uses a fine-tuned BART model trained on CNN/DailyMail dataset.
    
    **Model**: facebook/bart-large-cnn (fine-tuned)  
    **Training Data**: CNN/DailyMail  
    **Framework**: HuggingFace Transformers
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
