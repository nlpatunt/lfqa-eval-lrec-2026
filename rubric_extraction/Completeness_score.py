
from config.OpenRouter import OpenRouter
import math
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

class Completeness_score(object):

    def log_prob_extractor_1to5(self, content_logprobs):

        digit_set = {"1", "2", "3", "4", "5"}
        digit_logprobs = []

        if not content_logprobs:
            return digit_logprobs

        # take the first position's top_logprobs
        first_pos = content_logprobs[0].get("top_logprobs", [])
        for t in first_pos:
            tok = t.get("token", "").lstrip("Ġ \n\r\t")
            if tok in digit_set:
                digit_logprobs.append((int(tok), t.get("logprob")))

        return digit_logprobs

    def prob_weighted_score(self,logprobs):
        """
            G-Eval scoring function:

            score = Σ_i [ p(s_i) * s_i ]
            where
                p(s_i) = exp(ℓ(s_i)) / Σ_j exp(ℓ(s_j))

            - ℓ(s_i) = log probability of rating token s_i
            - s_i ∈ {1,2,3,4,5}

            This normalizes the log-probabilities into a distribution
            over the 5 possible scores, then computes the expected value.
        """
        probs = [(score, math.exp(lp)) for score, lp in logprobs] # Convert logprobs -> probs

        Z = sum(p for _, p in probs)  # normalization

        probs = [(s, p / Z) for s, p in probs]

        # GEVAL: Weighted sum
        score = sum(s * p for s, p in probs)

        return score, probs

        
    def compute(self,router,question,answer):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\geval_completeness_instructions.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()


        #question = "Why do old movies have that signature soft glow around the actors when up close?"
        #answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer)
      
        
        response,content_logprobs = router.get_response_geval_logprob(prompt)



        final_score, normalized_probs = self.prob_weighted_score(self.log_prob_extractor_1to5(content_logprobs))
        return final_score,self.extract_integers(response), round(final_score)
        #print(normalized_probs)
   
    def extract_integers(self, text: str) -> int:
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        else:
            return None  # or raise ValueError("No integer found")

    def compare_yes_g_in_one_function(self, completeness_scores, geval_scores):
        """
        Compare YESciEval (ints) vs G-Eval (floats).
        Prints Spearman, Kendall, Pearson, MAE, RMSE, QWK(rounded).
        Returns a dict with the same metrics.
        """
        # --- prepare data ---
        y_int = np.asarray(completeness_scores, dtype=int)
        g = np.asarray(geval_scores, dtype=float)
        if y_int.shape[0] != g.shape[0]:
            raise ValueError(f"Length mismatch: YES={y_int.shape[0]} vs G={g.shape[0]}")

        # --- rank-based (scale-invariant) ---
        spearman = spearmanr(y_int, g).correlation
        kendall = kendalltau(y_int, g).correlation
        print(spearman,kendall)

     
    def test_performance_geval_yescieval(self, router):
        file_path = r"C:\Users\rafid\Downloads\bioasq-dataset\all\original_synthesis\BioASQ_dataset_synthesis_per_model_evaluation_meta-llama-3.1-8b-instruct_clean.xlsx"
        # Read the Excel file
        df = pd.read_excel(file_path)



        # Pick one column to display (example: 'research_question')
        questions_column = "research_question"
        answers_column = "meta-llama-3.1-8b-instruct_synthesis"
        completeness_scores_column = "meta-llama-3.1-8b-instruct_meta-llama-3.1-8b-instruct_Completeness"


        questions = df[questions_column]
        answers = df[answers_column]
        completeness_scores = df[completeness_scores_column]
        geval_scores = []
        geval_scores_round = []
        geval_raw_scores = []

        print(len(questions))
        match_counter = 0
        for i in range(len(questions)):
            geval_score,response,geval_score_round = self.compute(router,questions[i],answers[i])
            geval_scores.append(geval_score)
            geval_scores_round.append(geval_score_round)
            geval_raw_scores.append(response)
            print(geval_score, completeness_scores[i])
            if (geval_score_round == completeness_scores[i]):
                match_counter += 1

        print(match_counter)
        print(list(completeness_scores))
        print(len(geval_scores))
        print(geval_scores)
        print(geval_scores_round)
        print(len(geval_scores_round))
        print(geval_raw_scores)
        print(len(geval_raw_scores))
        print("GEVAL scores")
        self.compare_yes_g_in_one_function(list(completeness_scores),geval_scores)
        print("GEVAL scores round")
        self.compare_yes_g_in_one_function(list(completeness_scores),geval_scores_round)
        print("GEVAL raw scores")
        self.compare_yes_g_in_one_function(list(completeness_scores),geval_raw_scores)


    