
from config.OpenRouter import OpenRouter

class Coherence_score(object):

    def log_prob_extractor_1to5(self, content_logprobs):

        digit_set = {"1", "2", "3", "4", "5"}
        digit_logprobs = []

        for t in content_logprobs or []:
            tok = t.get("token", "").lstrip("Ġ \n\r\t")  # strip common prefixes
            if tok in digit_set:
                digit_logprobs.append((int(tok), t.get("logprob")))

        return digit_logprobs

    def update_response_llm(self,router):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\geval_coherence_instructions.txt'

        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()


        question = "Why do old movies have that signature soft glow around the actors when up close?"
        answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer)
      
        
        response,content_logprobs = router.get_response_geval_logprob(prompt)
        print(self.log_prob_extractor_1to5(content_logprobs))
   