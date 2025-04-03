"""
utils/stream_handler.py
Stream handlers for LLM output in Streamlit
"""
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    """
    Stream handler for LLM output in Streamlit
    
    This handler captures tokens from an LLM and updates a Streamlit container
    in real-time, creating a streaming effect.
    """
    
    def __init__(self, container):
        """
        Initialize the stream handler
        
        Args:
            container: Streamlit container to update
        """
        self.container = container
        self.text = ""
        self.message_placeholder = container.empty()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Called when LLM produces a new token
        
        Args:
            token: New token from LLM
            kwargs: Additional arguments
        """
        self.text += token
        self.message_placeholder.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """
        Called when LLM generation ends
        
        Args:
            response: LLM response
            kwargs: Additional arguments
        """
        self.message_placeholder.markdown(self.text)


class BufferedStreamHandler(BaseCallbackHandler):
    """
    Buffered stream handler for LLM output in Streamlit
    
    This handler buffers tokens from an LLM and updates a Streamlit container
    at regular intervals, reducing UI updates for better performance.
    """
    
    def __init__(self, container, buffer_size=10):
        """
        Initialize the buffered stream handler
        
        Args:
            container: Streamlit container to update
            buffer_size: Number of tokens to buffer before updating UI
        """
        self.container = container
        self.text = ""
        self.buffer = ""
        self.buffer_size = buffer_size
        self.token_count = 0
        self.message_placeholder = container.empty()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Called when LLM produces a new token
        
        Args:
            token: New token from LLM
            kwargs: Additional arguments
        """
        self.buffer += token
        self.token_count += 1
        
        # Update UI when buffer is full
        if self.token_count >= self.buffer_size:
            self.text += self.buffer
            self.message_placeholder.markdown(self.text + "▌")
            self.buffer = ""
            self.token_count = 0
    
    def on_llm_end(self, response, **kwargs) -> None:
        """
        Called when LLM generation ends
        
        Args:
            response: LLM response
            kwargs: Additional arguments
        """
        # Flush remaining buffer
        self.text += self.buffer
        self.message_placeholder.markdown(self.text)


class ProgressStreamHandler(BaseCallbackHandler):
    """
    Stream handler with progress indicator for LLM output in Streamlit
    
    This handler updates a Streamlit container with tokens from an LLM and
    shows a progress bar to indicate completion status.
    """
    
    def __init__(self, container, progress_bar, estimated_tokens=1000):
        """
        Initialize the progress stream handler
        
        Args:
            container: Streamlit container to update
            progress_bar: Streamlit progress bar
            estimated_tokens: Estimated total number of tokens
        """
        self.container = container
        self.progress_bar = progress_bar
        self.estimated_tokens = estimated_tokens
        self.token_count = 0
        self.text = ""
        self.message_placeholder = container.empty()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Called when LLM produces a new token
        
        Args:
            token: New token from LLM
            kwargs: Additional arguments
        """
        self.text += token
        self.token_count += 1
        
        # Update progress bar
        progress = min(1.0, self.token_count / self.estimated_tokens)
        self.progress_bar.progress(progress)
        
        # Update text
        self.message_placeholder.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """
        Called when LLM generation ends
        
        Args:
            response: LLM response
            kwargs: Additional arguments
        """
        self.progress_bar.progress(1.0)
        self.message_placeholder.markdown(self.text)