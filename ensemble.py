from collections import defaultdict
from torch import nn
import torch
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers import AutoModelForSeq2SeqLM
from copy import deepcopy

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class EnsembleForConditionalGeneration(nn.Module, GenerationMixin):
    def __init__(self, models, device):
        super().__init__()
        self.models = [AutoModelForSeq2SeqLM.from_pretrained(i).to(device) for i in models]
        self.encoder = EnsembleEncoder(self.models)
        assert len(set(type(i) for i in self.models)) == 1, "All models have to have the same type"
        self.config = self.models[0].config
        self.device = device

    def forward(self, *args, **kwargs):
        output = Seq2SeqLMOutput()
        for i, model in enumerate(self.models):
            kwargs_copy = deepcopy(kwargs) 
            kwargs_copy["encoder_outputs"]["last_hidden_state"] = kwargs["encoder_outputs"]["last_hidden_state"][:, i, ...]
            logits = [model(*args, **kwargs_copy)["logits"] for model in self.models][:1]

        output["logits"] = torch.mean(torch.stack(logits, dim=0), dim=0)

        return output

    def half(self):
        self.models = [i.half() for i in self.models]
        return self

    def get_encoder(self):
        return self.encoder

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


class EnsembleEncoder(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(
        self,
        *args, **kwargs
    ):
        outputs = defaultdict(list)
        for model in self.models:
            output = model.get_encoder()(*args, **kwargs)
            for k, v in output.items():
                outputs[k].append(v)

        for k, v in outputs.items():
            try:
                outputs[k] = torch.stack(v, dim=1)
            except TypeError:
                outputs[k] = v

        return BaseModelOutput(**outputs)