import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from text_evaluator import TextEvaluator


class ObsidianExperimentLogger:
    """
    Logger class to save text readability experiment results to Obsidian notes.
    
    This class provides methods to save experiment results as markdown files
    in your Obsidian vault, with proper formatting and metadata.
    """
    
    def __init__(self, vault_path: str, experiments_folder: str = "Experiments"):
        """
        Initialize the Obsidian logger.
        
        Args:
            vault_path (str): Path to your Obsidian vault
            experiments_folder (str): Subfolder name for experiments (default: "Experiments")
        """
        self.vault_path = vault_path
        self.experiments_folder = experiments_folder
        self.experiments_path = os.path.join(vault_path, experiments_folder)
        
        # Create experiments folder if it doesn't exist
        os.makedirs(self.experiments_path, exist_ok=True)
        
        # Initialize text evaluator
        self.evaluator = TextEvaluator()
    
    def create_experiment_note(self, 
                             experiment_name: str,
                             text_sample: str,
                             analysis_results: Dict[str, Any],
                             tags: Optional[list] = None,
                             notes: Optional[str] = None) -> str:
        """
        Create a new experiment note in Obsidian.
        
        Args:
            experiment_name (str): Name of the experiment
            text_sample (str): The text that was analyzed
            analysis_results (Dict): Results from TextEvaluator
            tags (list, optional): Tags to add to the note
            notes (str, optional): Additional notes or observations
            
        Returns:
            str: Path to the created note file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{experiment_name}_{timestamp}.md"
        filepath = os.path.join(self.experiments_path, filename)
        
        # Prepare tags
        if tags is None:
            tags = ["experiment", "readability", "text-analysis"]
        
        # Create markdown content
        content = self._format_experiment_note(
            experiment_name, text_sample, analysis_results, tags, notes, timestamp
        )
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Experiment saved to: {filepath}")
        return filepath
    
    def _format_experiment_note(self, 
                               experiment_name: str,
                               text_sample: str,
                               results: Dict[str, Any],
                               tags: list,
                               notes: Optional[str],
                               timestamp: str) -> str:
        """Format the experiment data as markdown."""
        
        # Create frontmatter
        frontmatter = "---\n"
        frontmatter += f"title: {experiment_name}\n"
        frontmatter += f"date: {timestamp}\n"
        frontmatter += f"tags: {tags}\n"
        frontmatter += "type: experiment\n"
        frontmatter += "---\n\n"
        
        # Main content
        content = frontmatter
        content += f"# {experiment_name}\n\n"
        content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Text sample
        content += "## 📝 Text Sample\n\n"
        content += "```\n"
        content += text_sample[:500] + ("..." if len(text_sample) > 500 else "")
        content += "\n```\n\n"
        
        # Text statistics
        if 'text_statistics' in results:
            stats = results['text_statistics']
            content += "## 📊 Text Statistics\n\n"
            content += f"- **Sentences:** {stats['sentence_count']}\n"
            content += f"- **Words:** {stats['word_count']}\n"
            content += f"- **Characters:** {stats['character_count']}\n"
            content += f"- **Syllables:** {stats['syllable_count']}\n"
            content += f"- **Polysyllable words:** {stats['polysyllable_count']}\n"
            content += f"- **Difficult words:** {stats['difficult_words']}\n\n"
        
        # Grade level indices
        if 'grade_level_indices' in results:
            grades = results['grade_level_indices']
            content += "## 🎓 Grade Level Indices\n\n"
            content += f"- **Flesch-Kincaid Grade:** {grades['flesch_kincaid_grade']}\n"
            content += f"- **Gunning Fog Index:** {grades['gunning_fog']}\n"
            content += f"- **Automated Readability Index:** {grades['automated_readability_index']}\n"
            content += f"- **Coleman-Liau Index:** {grades['coleman_liau_index']}\n"
            content += f"- **Dale-Chall Score:** {grades['dale_chall_readability_score']}\n\n"
        
        # Readability scores
        if 'readability_scores' in results:
            scores = results['readability_scores']
            content += "## 📖 Readability Scores\n\n"
            content += f"- **Flesch Reading Ease:** {scores['flesch_reading_ease']} *(0-100, higher = easier)*\n"
            content += f"- **Linsear Write Formula:** {scores['linsear_write_formula']}\n"
            content += f"- **Spache Readability:** {scores['spache_readability']}\n"
            content += f"- **McAlpine EFLAW:** {scores['mcalpine_eflaw']}\n\n"
        
        # Spanish scores if available
        if 'spanish_scores' in results and results['spanish_scores']:
            spanish = results['spanish_scores']
            content += "## 🇪🇸 Spanish Readability Scores\n\n"
            content += f"- **Fernández-Huerta:** {spanish['fernandez_huerta']}\n"
            content += f"- **Szigriszt-Pazos:** {spanish['szigriszt_pazos']}\n"
            content += f"- **Gutiérrez de Polini:** {spanish['gutierrez_polini']}\n"
            content += f"- **Crawford:** {spanish['crawford']}\n\n"
        
        # Reading time
        if 'reading_time' in results:
            time_info = results['reading_time']
            content += "## ⏱️ Reading Time\n\n"
            content += f"- **Estimated time:** {time_info['reading_time_minutes']} minutes\n"
            content += f"- **({time_info['reading_time_seconds']} seconds)**\n\n"
        
        # Additional notes
        if notes:
            content += "## 📋 Notes\n\n"
            content += notes + "\n\n"
        
        # Raw data section
        content += "## 🔧 Raw Data\n\n"
        content += "```json\n"
        content += json.dumps(results, indent=2, ensure_ascii=False)
        content += "\n```\n\n"
        
        # Links section
        content += "## 🔗 Related\n\n"
        content += "- [[Text Analysis Methods]]\n"
        content += "- [[Readability Metrics Comparison]]\n"
        content += "- [[Thesis Experiments]]\n"
        
        return content
    
    def run_experiment_and_log(self, 
                              experiment_name: str,
                              text_sample: str,
                              include_spanish: bool = False,
                              tags: Optional[list] = None,
                              notes: Optional[str] = None) -> str:
        """
        Run a complete text analysis experiment and log to Obsidian.
        
        Args:
            experiment_name (str): Name of the experiment
            text_sample (str): Text to analyze
            include_spanish (bool): Include Spanish-specific metrics
            tags (list, optional): Tags for the note
            notes (str, optional): Additional observations
            
        Returns:
            str: Path to the created note
        """
        print(f"🔬 Running experiment: {experiment_name}")
        
        # Run analysis
        results = self.evaluator.evaluate_text_comprehensive(text_sample, include_spanish)
        
        # Log to Obsidian
        note_path = self.create_experiment_note(
            experiment_name, text_sample, results, tags, notes
        )
        
        return note_path
    
    def create_experiment_summary(self, experiment_files: list) -> str:
        """
        Create a summary note linking multiple experiments.
        
        Args:
            experiment_files (list): List of experiment note filenames
            
        Returns:
            str: Path to the summary note
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"Experiment_Summary_{timestamp}.md"
        filepath = os.path.join(self.experiments_path, filename)
        
        content = "---\n"
        content += f"title: Experiment Summary {timestamp}\n"
        content += f"date: {timestamp}\n"
        content += "tags: [summary, experiments, readability]\n"
        content += "type: summary\n"
        content += "---\n\n"
        
        content += f"# Experiment Summary - {timestamp}\n\n"
        content += "## 📋 Experiments Conducted\n\n"
        
        for exp_file in experiment_files:
            exp_name = os.path.splitext(exp_file)[0]
            content += f"- [[{exp_name}]]\n"
        
        content += "\n## 📊 Analysis\n\n"
        content += "*Add your comparative analysis here*\n\n"
        content += "## 🎯 Conclusions\n\n"
        content += "*Add your conclusions here*\n\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Summary created: {filepath}")
        return filepath


# Example usage
if __name__ == "__main__":
    # Configure your Obsidian vault path
    VAULT_PATH = "/Users/santiago/Documents/Obsidian Vault"  # Update this path!
    
    # Create logger
    logger = ObsidianExperimentLogger(VAULT_PATH)
    
    # Sample texts for testing
    english_text = """
    The rapid advancement of artificial intelligence has revolutionized numerous industries. 
    Machine learning algorithms can process vast amounts of data with unprecedented efficiency. 
    These sophisticated computational tools enable automation of complex decision-making processes 
    that were previously impossible to implement effectively.
    """
    
    spanish_text = """
    La inteligencia artificial está transformando múltiples industrias mediante algoritmos 
    sofisticados y técnicas de aprendizaje automático. Estas herramientas computacionales 
    avanzadas permiten automatizar procesos complejos de toma de decisiones.
    """
    
    # Run experiments
    print("Running English text experiment...")
    logger.run_experiment_and_log(
        experiment_name="English_AI_Text_Analysis",
        text_sample=english_text,
        tags=["english", "ai-text", "technical"],
        notes="Testing readability of AI-related technical text in English."
    )
    
    print("\nRunning Spanish text experiment...")
    logger.run_experiment_and_log(
        experiment_name="Spanish_AI_Text_Analysis", 
        text_sample=spanish_text,
        include_spanish=True,
        tags=["spanish", "ai-text", "technical"],
        notes="Testing readability of AI-related technical text in Spanish with Spanish-specific metrics."
    )
