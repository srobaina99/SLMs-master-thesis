"""
Response formatting utilities for cleaning LLM responses before evaluation.
Removes formatting elements that might interfere with text complexity scoring.
"""

import re


class ResponseFormatter:
    """
    Utility class for cleaning and formatting LLM responses before evaluation.
    
    Removes elements like emojis, arrows, special formatting that might interfere
    with readability metrics while preserving the core text content.
    """
    
    def __init__(self):
        """Initialize the formatter with predefined patterns."""
        # Emoji patterns (Unicode ranges for common emojis)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", 
            flags=re.UNICODE
        )
        
        # Arrow and bullet patterns
        self.arrow_pattern = re.compile(r'[→←↑↓↔↕➜➤➡⬅⬆⬇⬌⬍⟶⟵⟷⟸⟹⟺]')
        self.bullet_pattern = re.compile(r'[•·‣⁃▪▫◦‰‱]')
        
        # Special formatting characters
        self.special_chars_pattern = re.compile(r'[★☆✓✗✘✔✖⚡⚠⭐🔥💡📝📊📈📉]')
        
        # Multiple spaces/tabs/newlines
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Common markdown-style formatting
        self.markdown_pattern = re.compile(r'[*_`#]+')
        
        # Parenthetical expressions with symbols
        self.symbol_parentheses_pattern = re.compile(r'\([^a-zA-Z0-9\s,.-]+\)')
    
    def remove_emojis(self, text: str) -> str:
        """Remove all emojis from text."""
        return self.emoji_pattern.sub('', text)
    
    def remove_arrows_and_bullets(self, text: str) -> str:
        """Remove arrow symbols and bullet points."""
        text = self.arrow_pattern.sub('', text)
        text = self.bullet_pattern.sub('', text)
        return text
    
    def remove_special_formatting(self, text: str) -> str:
        """Remove special formatting characters."""
        return self.special_chars_pattern.sub('', text)
    
    def remove_markdown_formatting(self, text: str) -> str:
        """Remove basic markdown formatting characters."""
        return self.markdown_pattern.sub('', text)
    
    def remove_symbol_parentheses(self, text: str) -> str:
        """Remove parentheses that contain only symbols."""
        return self.symbol_parentheses_pattern.sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize multiple spaces, tabs, and newlines to single spaces."""
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        # Strip leading/trailing whitespace
        return text.strip()
    
    def clean_response_for_evaluation(self, response: str) -> str:
        """
        Comprehensive cleaning of LLM response for text evaluation.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Cleaned text with only words and basic punctuation
        """
        if not response or not response.strip():
            return ""
        
        cleaned_text = response
        
        # Apply all cleaning steps in sequence
        cleaned_text = self.remove_emojis(cleaned_text)
        cleaned_text = self.remove_arrows_and_bullets(cleaned_text)
        cleaned_text = self.remove_special_formatting(cleaned_text)
        cleaned_text = self.remove_markdown_formatting(cleaned_text)
        cleaned_text = self.remove_symbol_parentheses(cleaned_text)
        cleaned_text = self.normalize_whitespace(cleaned_text)
        
        return cleaned_text
    
    def clean_preserving_structure(self, response: str) -> str:
        """
        Light cleaning that preserves text structure but removes interfering elements.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Lightly cleaned text preserving paragraphs and sentences
        """
        if not response or not response.strip():
            return ""
        
        cleaned_text = response
        
        # Only remove the most problematic elements
        cleaned_text = self.remove_emojis(cleaned_text)
        cleaned_text = self.remove_special_formatting(cleaned_text)
        cleaned_text = self.remove_symbol_parentheses(cleaned_text)
        
        # Normalize excessive whitespace but preserve line breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces/tabs to single space
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Multiple newlines to double
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text


# Convenience functions for quick usage
def clean_for_evaluation(response: str) -> str:
    """Quick function to clean response for evaluation."""
    formatter = ResponseFormatter()
    return formatter.clean_response_for_evaluation(response)


def clean_preserving_structure(response: str) -> str:
    """Quick function to lightly clean response while preserving structure."""
    formatter = ResponseFormatter()
    return formatter.clean_preserving_structure(response)


# Example usage and testing
if __name__ == "__main__":
    # Test cases with various formatting elements
    test_responses = [
        "Hello! 😊 This is a simple response with emojis 🎉 and arrows → pointing here.",
        "• First point with bullet\n• Second point ✓\n→ Arrow pointing to conclusion",
        "**Bold text** with *italics* and `code` formatting that needs cleaning.",
        "Response with (✓) symbols in parentheses and ⚡ special chars throughout.",
        "Multiple    spaces   and\n\n\n\nexcessive newlines need normalization.",
        "Complex response 🔥 with → arrows, • bullets, **formatting**, and (⭐) symbols mixed together!"
    ]
    
    formatter = ResponseFormatter()
    
    print("=== Response Formatting Test ===\n")
    
    for i, response in enumerate(test_responses, 1):
        print(f"Test {i}:")
        print(f"Original: {response}")
        print(f"Cleaned:  {formatter.clean_response_for_evaluation(response)}")
        print(f"Light:    {formatter.clean_preserving_structure(response)}")
        print("-" * 80)
