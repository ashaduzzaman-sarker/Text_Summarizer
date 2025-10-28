# ============================================================================
# gradio_app.py (Gradio Interface)
# ============================================================================
"""Gradio web interface for text summarization."""

import gradio as gr
from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.logging.logger import logger


# Initialize predictor
try:
    predictor = PredictionPipeline()
    logger.info("Prediction pipeline initialized for Gradio")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    predictor = None


def summarize_text(
    text: str,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4
) -> tuple:
    """Summarize text and return summary with statistics.
    
    Args:
        text: Input article text
        max_length: Maximum summary length
        min_length: Minimum summary length
        num_beams: Beam search width
        
    Returns:
        Tuple of (summary, statistics)
    """
    if not text or len(text.strip()) < 50:
        return "⚠️ Please enter at least 50 characters of text.", ""
    
    if predictor is None:
        return "❌ Model not loaded. Please check logs.", ""
    
    try:
        # Generate summary
        summary = predictor.predict(
            text=text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams
        )
        
        # Calculate statistics
        original_length = len(text)
        summary_length = len(summary)
        compression_ratio = round(summary_length / original_length * 100, 2)
        
        stats = f"""
        📊 **Statistics:**
        - Original Length: {original_length} characters
        - Summary Length: {summary_length} characters
        - Compression Ratio: {compression_ratio}%
        - Word Count: {len(text.split())} → {len(summary.split())} words
        """
        
        return summary, stats
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"❌ Error: {str(e)}", ""


# Sample articles for demo
EXAMPLES = [
    [
        """(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze medal. Bolt has now collected eight gold medals at world championships, having previously won the 100m and 200m races in the Russian capital. "I'm proud of myself and I'll continue to work to dominate for as long as possible," Bolt said, having previously expressed his intention to carry on until the 2016 Rio Olympics. Victory was never in doubt once he got the baton safely in hand from Ashmeade, while Gatlin and the United States third leg runner Rakieem Salaam failed to get the baton away cleanly. Earlier, Jamaica's women underlined their dominance in the sprint events by winning the 4x100m relay gold, anchored by Shelly-Ann Fraser-Pryce, who like Bolt was completing a triple.""",
        128, 30, 4
    ],
    [
        """Microsoft has announced a major update to its Windows operating system, introducing new AI-powered features designed to enhance productivity and user experience. The update, scheduled for release next month, includes an advanced virtual assistant capable of understanding natural language commands and performing complex tasks across multiple applications. Additionally, the update brings improved security features, better battery management for laptops, and a redesigned user interface that adapts to individual usage patterns. Industry analysts predict this could be Microsoft's most significant update in years, potentially reshaping how users interact with their computers.""",
        128, 30, 4
    ]
]


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Text Summarizer") as demo:
    gr.Markdown(
        """
        # 📝 Text Summarization
        ### Powered by Fine-tuned BART Model
        
        Enter a news article or long text, and get an AI-generated summary instantly!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Paste your article here (minimum 50 characters)...",
                lines=10,
                max_lines=20
            )
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=30,
                    maximum=512,
                    value=128,
                    step=10,
                    label="Max Summary Length",
                    info="Maximum length of generated summary"
                )
                
                min_length_slider = gr.Slider(
                    minimum=10,
                    maximum=256,
                    value=30,
                    step=5,
                    label="Min Summary Length",
                    info="Minimum length of generated summary"
                )
            
            num_beams_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                label="Beam Search Width",
                info="Higher = better quality but slower (4 recommended)"
            )
            
            with gr.Row():
                summarize_btn = gr.Button("✨ Generate Summary", variant="primary", size="lg")
                clear_btn = gr.ClearButton([text_input], value="🗑️ Clear")
        
        with gr.Column(scale=2):
            # Output section
            summary_output = gr.Textbox(
                label="Generated Summary",
                lines=8,
                max_lines=15
            )
            
            stats_output = gr.Markdown(
                label="Statistics"
            )
    
    # Examples section
    gr.Markdown("### 📚 Try These Examples:")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, max_length_slider, min_length_slider, num_beams_slider],
        outputs=[summary_output, stats_output],
        fn=summarize_text,
        cache_examples=False
    )
    
    # Event handlers
    summarize_btn.click(
        fn=summarize_text,
        inputs=[text_input, max_length_slider, min_length_slider, num_beams_slider],
        outputs=[summary_output, stats_output]
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model:** Fine-tuned BART on CNN/DailyMail Dataset  
        **Performance:** ROUGE-1: 0.42 | ROUGE-2: 0.19 | ROUGE-L: 0.30
        """
    )


# Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # False,  # Set True to create public link
        show_error=True
    )
