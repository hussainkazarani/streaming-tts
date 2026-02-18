import re
import logging

# Initialize the module-specific logger
logger = logging.getLogger(__name__)

def split_text(text: str, max_words: int = 400) -> list[str]:
    """
    Splits a large string of text into manageable chunks to prevent 
    the TTS engine from exceeding its maximum context window.
    
    The algorithm prioritizes natural pauses by splitting on paragraphs first. 
    If a paragraph is still too long, it falls back to splitting on sentence 
    boundaries (periods, exclamation marks, question marks).
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = []
    
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            # Paragraph is short enough; keep it as one chunk
            chunks.append(para)
        else:
            # Paragraph is too long; split by sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            
            for s in sentences:
                if not current:
                    current = s
                elif len((current + " " + s).split()) <= max_words:
                    current += " " + s
                else:
                    chunks.append(current)
                    current = s
                    
            if current:
                chunks.append(current)
                
    logger.debug(f"Split incoming text into {len(chunks)} processing chunk(s).")
    return chunks