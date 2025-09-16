"""
Standardized prompts for English learning conversation experiments.
Contains various scenarios and system prompts for testing LLM teaching capabilities.
"""

from typing import Dict, List


class EnglishLearningPrompts:
    """Collection of standardized prompts for English learning experiments."""
    
    # System prompts for different teaching styles
    SYSTEM_PROMPTS = {
        'basic_teacher': (
            "You are an English teacher helping a beginner student. "
            "Use simple words and short sentences. Be patient and encouraging."
        ),
        
        'detailed_teacher': (
            "You are an experienced English teacher. Provide clear explanations "
            "with examples. Use simple vocabulary but give thorough answers."
        ),
        
        'conversational_teacher': (
            "You are a friendly English conversation partner. Help the student "
            "learn through natural dialogue. Keep responses engaging but simple."
        ),
        
        'grammar_focused': (
            "You are an English grammar teacher. Focus on correct grammar usage "
            "and explain rules clearly. Use beginner-friendly language."
        ),
        
        'vocabulary_focused': (
            "You are an English vocabulary teacher. Help students learn new words "
            "with clear definitions and simple examples."
        )
    }
    
    # Vocabulary and grammar questions
    VOCABULARY_QUESTIONS = [
        "What does the word 'library' mean?",
        "Can you explain the difference between 'big' and 'large'?",
        "What is a 'restaurant'? Can you give me an example?",
        "What does 'happy' mean? Can you use it in a sentence?",
        "Explain what 'weather' means with simple words.",
        "What is the difference between 'house' and 'home'?",
        "Can you tell me what 'family' means?",
        "What does 'friend' mean? How do you make friends?",
        "Explain the word 'school' to me.",
        "What is 'food'? Can you name some types of food?"
    ]
    
    # Grammar and usage questions
    GRAMMAR_QUESTIONS = [
        "When do I use 'a' and when do I use 'an'?",
        "What is the difference between 'I am' and 'I have'?",
        "How do I ask for directions in English?",
        "When should I say 'please' and 'thank you'?",
        "What is the difference between 'this' and 'that'?",
        "How do I make a sentence negative in English?",
        "When do I use 'is' and when do I use 'are'?",
        "How do I ask questions in English?",
        "What is the difference between 'he' and 'she'?",
        "How do I talk about things I like and don't like?"
    ]
    
    # Conversational scenarios
    CONVERSATION_SCENARIOS = [
        "How do I introduce myself to someone new?",
        "What should I say when I meet someone for the first time?",
        "How do I order food at a restaurant?",
        "What do I say when someone asks 'How are you?'",
        "How do I ask for help when I don't understand something?",
        "What should I say when I want to buy something at a store?",
        "How do I make small talk about the weather?",
        "What do I say when I want to make plans with a friend?",
        "How do I politely disagree with someone?",
        "What should I say when I'm late for an appointment?"
    ]
    
    # Cultural and practical questions
    CULTURAL_QUESTIONS = [
        "What are some common greetings in English-speaking countries?",
        "How do people celebrate birthdays in English-speaking countries?",
        "What is considered polite behavior when visiting someone's home?",
        "How do people typically spend their weekends?",
        "What are some popular foods in English-speaking countries?",
        "How do people usually commute to work or school?",
        "What are some common hobbies people have?",
        "How do people typically celebrate holidays?",
        "What is the school system like in English-speaking countries?",
        "How do people make friends in English-speaking countries?"
    ]
    
    # Error correction scenarios (student makes common mistakes)
    ERROR_CORRECTION_PROMPTS = [
        "I am very good in English. Is this sentence correct?",
        "Yesterday I go to the store. Can you help me fix this sentence?",
        "She don't like pizza. What's wrong with this sentence?",
        "I have 25 years old. Is this the right way to say my age?",
        "Can you learn me English? Is this sentence correct?",
        "I am agree with you. How should I say this correctly?",
        "I am boring in class. What's the correct way to express this?",
        "I have been to London yesterday. Is this sentence right?",
        "She is more tall than me. How should I compare heights?",
        "I am interesting in music. Is this the correct way to say it?"
    ]
    
    @classmethod
    def get_all_prompts(cls) -> List[str]:
        """Get all prompts combined into a single list."""
        all_prompts = []
        all_prompts.extend(cls.VOCABULARY_QUESTIONS)
        all_prompts.extend(cls.GRAMMAR_QUESTIONS)
        all_prompts.extend(cls.CONVERSATION_SCENARIOS)
        all_prompts.extend(cls.CULTURAL_QUESTIONS)
        all_prompts.extend(cls.ERROR_CORRECTION_PROMPTS)
        return all_prompts
    
    @classmethod
    def get_prompts_by_category(cls) -> Dict[str, List[str]]:
        """Get prompts organized by category."""
        return {
            'vocabulary': cls.VOCABULARY_QUESTIONS,
            'grammar': cls.GRAMMAR_QUESTIONS,
            'conversation': cls.CONVERSATION_SCENARIOS,
            'cultural': cls.CULTURAL_QUESTIONS,
            'error_correction': cls.ERROR_CORRECTION_PROMPTS
        }
    
    @classmethod
    def get_basic_test_set(cls) -> List[str]:
        """Get a smaller set of prompts for quick testing."""
        return [
            "What does the word 'library' mean?",
            "When do I use 'a' and when do I use 'an'?",
            "How do I introduce myself to someone new?",
            "What are some common greetings in English-speaking countries?",
            "I am very good in English. Is this sentence correct?"
        ]
    
    @classmethod
    def get_system_prompt_variations(cls) -> Dict[str, str]:
        """Get all available system prompt variations."""
        return cls.SYSTEM_PROMPTS.copy()


# Predefined experiment configurations for common test scenarios
STANDARD_EXPERIMENT_CONFIGS = {
    'basic_teaching': {
        'system_prompt': EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
        'prompts': EnglishLearningPrompts.VOCABULARY_QUESTIONS[:5],
        'description': 'Basic vocabulary teaching with simple system prompt'
    },
    
    'grammar_focus': {
        'system_prompt': EnglishLearningPrompts.SYSTEM_PROMPTS['grammar_focused'],
        'prompts': EnglishLearningPrompts.GRAMMAR_QUESTIONS[:5],
        'description': 'Grammar-focused teaching approach'
    },
    
    'conversation_practice': {
        'system_prompt': EnglishLearningPrompts.SYSTEM_PROMPTS['conversational_teacher'],
        'prompts': EnglishLearningPrompts.CONVERSATION_SCENARIOS[:5],
        'description': 'Conversational English practice'
    },
    
    'error_correction': {
        'system_prompt': EnglishLearningPrompts.SYSTEM_PROMPTS['detailed_teacher'],
        'prompts': EnglishLearningPrompts.ERROR_CORRECTION_PROMPTS[:5],
        'description': 'Error correction and explanation'
    },
    
    'quick_test': {
        'system_prompt': EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
        'prompts': EnglishLearningPrompts.get_basic_test_set(),
        'description': 'Quick test with diverse prompt types'
    }
}
