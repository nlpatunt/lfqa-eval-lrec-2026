
import math
import textstat

class Easy_to_understand_score(object):
    def log_prob_extractor_1to5(self, content_logprobs):

        digit_set = {"1", "2", "3"}
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
            - s_i ∈ {1,2,3}

            This normalizes the log-probabilities into a distribution
            over the 5 possible scores, then computes the expected value.
        """
        probs = [(score, math.exp(lp)) for score, lp in logprobs] # Convert logprobs -> probs

        Z = sum(p for _, p in probs)  # normalization

        probs = [(s, p / Z) for s, p in probs]

        # GEVAL: Weighted sum
        score = sum(s * p for s, p in probs)

        return score, probs

        
    def calculate_score_fluency_textstat(self,router):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\geval_fluency_instructions.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()


        question = "Why do old movies have that signature soft glow around the actors when up close?"
        answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer)
      
        
        response,content_logprobs = router.get_response_geval_logprob(prompt)



        final_score, normalized_probs = self.prob_weighted_score(self.log_prob_extractor_1to5(content_logprobs))
        print(final_score)
        print(normalized_probs)
        print(self.easy_to_understand_readability(answer))
   


    def easy_to_understand_readability(self,text: str) -> dict:
        """
        Easy-to-understand score using ONLY textstat readability metrics.
        """

        # --- Raw metrics (from textstat) ---
        fre  = textstat.flesch_reading_ease(text)                 
        fkgl = textstat.flesch_kincaid_grade(text)                
        dc   = textstat.dale_chall_readability_score(text)        
        smog = textstat.smog_index(text)                          
        gf   = textstat.gunning_fog(text)                         
        asl  = textstat.avg_sentence_length(text)                 

        # --- Normalize each metric into [0,1] ---
        # Clamp values manually so they don't fall outside [0,1].

        # Flesch Reading Ease: higher = better, expected range 0..100
        n_fre = (fre - 0.0) / (100.0 - 0.0)
        n_fre = max(0.0, min(1.0, n_fre))

        # Flesch–Kincaid Grade Level: lower = better, typical range 4..16
        n_fkgl = 1.0 - (fkgl - 4.0) / (16.0 - 4.0)
        n_fkgl = max(0.0, min(1.0, n_fkgl))

        # Dale–Chall: lower = better, typical range 5..10
        n_dc = 1.0 - (dc - 5.0) / (10.0 - 5.0)
        n_dc = max(0.0, min(1.0, n_dc))

        # SMOG Index: lower = better, typical range 5..18
        n_smog = 1.0 - (smog - 5.0) / (18.0 - 5.0)
        n_smog = max(0.0, min(1.0, n_smog))

        # Gunning Fog: lower = better, typical range 6..20
        n_gf = 1.0 - (gf - 6.0) / (20.0 - 6.0)
        n_gf = max(0.0, min(1.0, n_gf))

        # Avg sentence length: lower = better, typical range 10..35
        n_asl = 1.0 - (asl - 10.0) / (35.0 - 10.0)
        n_asl = max(0.0, min(1.0, n_asl))

        # --- Weighted combination ---
        score01 = (
            0.40 * n_fre +
            0.20 * n_fkgl +
            0.15 * n_dc +
            0.10 * n_smog +
            0.10 * n_gf +
            0.05 * n_asl
        )
        score_0_100 = round(100.0 * score01, 2)

        return {
            "score_0_100": score_0_100,
            "components": {
                "flesch_reading_ease": fre,
                "fk_grade_level": fkgl,
                "dale_chall": dc,
                "smog": smog,
                "gunning_fog": gf,
                "avg_sentence_length": asl,
                "norms_0_1": {
                    "n_flesch": n_fre,
                    "n_fkgl": n_fkgl,
                    "n_dale_chall": n_dc,
                    "n_smog": n_smog,
                    "n_gunning_fog": n_gf,
                    "n_avg_sentence_length": n_asl
                }
            }
        }