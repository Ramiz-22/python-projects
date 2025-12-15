"""Basic content moderation service"""
import re

# Basic inappropriate content keywords (expandable)
INAPPROPRIATE_KEYWORDS = [
    "spam", "scam", "phishing",
    # Add more keywords as needed
]


class ContentFilter:
    def __init__(self):
        self.keywords = INAPPROPRIATE_KEYWORDS
        
    def is_appropriate(self, text: str) -> bool:
        """
        Check if text is appropriate
        Returns True if appropriate, False otherwise
        """
        text_lower = text.lower()
        
        # Check for inappropriate keywords
        for keyword in self.keywords:
            if keyword in text_lower:
                return False
        
        # Check for excessive repetition (spam indicator)
        if self._has_excessive_repetition(text):
            return False
            
        return True
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Detect excessive character or word repetition"""
        # Check for same character repeated more than 10 times
        if re.search(r'(.)\1{10,}', text):
            return True
        
        # Check for same word repeated more than 5 times in a row
        words = text.split()
        if len(words) > 5:
            for i in range(len(words) - 5):
                if len(set(words[i:i+5])) == 1:
                    return True
        
        return False
    
    def filter_text(self, text: str) -> str:
        """
        Filter text and return cleaned version
        Returns original text if appropriate, sanitized version otherwise
        """
        if self.is_appropriate(text):
            return text
        
        # For MVP, just return a warning message
        return "[Content filtered for inappropriate content]"


# Singleton instance
content_filter = ContentFilter()
