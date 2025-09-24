# -*- coding: utf-8 -*-
import spacy

class Conciseness_score:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def lexical_density(self, text: str) -> float:
       
        #Calculate lexical density = content words / total words
        #Content words: "NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "X"
        #"PROPN",  # Proper nouns (names, e.g., "Python")  
        #NUM",    # Numbers (quantities, e.g., "2025")  
        #"X"       # Other/unknown tokens (slang/typos, e.g., "LOL")  
        
        doc = self.nlp(text) #Produces a Doc object with tokens, POS tags, etc.
        content_words = [token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "X"]
]
        total_words = [token for token in doc if token.is_alpha or token.pos_ in ["X", "SYM", "NUM"]]

        #Handle division by zero
        if not total_words:
            return 0.0
        return len(content_words) / len(total_words)

    def score(self, text: str) -> float:
        
        #Return a normalized conciseness score (0–1).
        #0 = not concise, 1 = very concise.
       
        return self.lexical_density(text)

if __name__ == "__main__":
    scorer = Conciseness_score()
    answer_a = (
        "Honestly I don't get why people keep asking the same question over and over again on this sub. "
        "Like dude just google it or check the docs first, it's literally the first result. "
        "Not trying to be rude but come on, it's 2025 and folks still don't know how to search properly smh."
    )
    answer_b = (
        "To fix this error you should reinstall Python, then set the PATH variable correctly in your system settings. "
        "After that, restart your terminal and verify by running python --version. "
        "If the issue persists, check whether multiple versions are installed and remove the conflicting one."
    )
    print("Answer A Conciseness:", scorer.score(answer_a))
    print("Answer B Conciseness:", scorer.score(answer_b))
