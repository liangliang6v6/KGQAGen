from src.graph_constrained_decoding import GraphConstrainedDecoding
from .base_hf_causal_model import HfCausalModel

class GraphConstrainedDecodingModel(HfCausalModel):
    def __init__(self, args):
        super().__init__(args)
    
    def generate_sentence(self, llm_input, trie, start_token_ids = None, end_token_ids = None, enable_constrained_by_default = True):
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, enable_constrained_by_default)
        try:
            res = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                generation_config=self.generation_cfg,
                prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(e)
            return None
        response = []
        if len(res.sequences) == 1:
            return self.tokenizer.decode(res.sequences[0][input_ids.shape[1]:],skip_special_tokens=True)
        for r in res.sequences:
            response.append(self.tokenizer.decode(r[input_ids.shape[1]:], 
          skip_special_tokens=True))
        return response
        
