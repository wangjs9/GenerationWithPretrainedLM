# coding: utf-8
import json
import logging
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig
from transformers.models.blenderbot_small import BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from utils.inputter_utils import Inputter
import nltk

sents_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class BaseModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.tokenizer = None

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))


class BlenderbotSmall(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            return_dict=None,
            validation=False,
            **kwargs
    ):
        assert (self.training or validation) == (labels is not None)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:  # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation:  # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training:  # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else:  # validation
            assert not self.training
            return loss, label_size

    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            return_dict=None,
            **kwargs
    ):
        assert not self.training
        assert decoder_input_ids.size(1) == 1

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        outputs = generations[:, decoder_input_ids.size(1):]

        return outputs


class ChatModel(object):

    def __init__(self,
                 pretrained_model_path,
                 max_src_turn=20,
                 max_input_length=256,
                 max_decode_length=64,
                 min_decode_length=1,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 num_beams=1,
                 repetition_penalty=1.0,
                 no_repeat_ngram_size=1,
                 use_gpu=False,
                 checkpoint=None,
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        logging.info("Loading pretrained tokenizer and model from {}".format(pretrained_model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.model = BlenderbotSmall.from_pretrained(pretrained_model_path)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        logging.info("Loaded")

        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        self.inputter = Inputter(tokenizer=self.tokenizer,
                                 max_src_turn=max_src_turn,
                                 max_input_length=max_input_length,
                                 max_decode_length=max_decode_length
                                 )

        self.generation_kwargs = {
            'max_length': max_decode_length,
            'min_length': min_decode_length,
            'do_sample': True if (top_k > 0 or top_p < 1) else False,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'num_beams': num_beams,
            'repetition_penalty': repetition_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'pad_token_id': pad,
            'bos_token_id': bos,
            'eos_token_id': eos,
        }

    def interact(self, dialog_context):
        """
        dialog_context: all usr-sys utterances. A dict with a list of dicts
        """
        logging.info("All input: {}".format(dialog_context))

        if "dialog" not in dialog_context:
            raise KeyError("The key `dialog` not in `dialog_context`!")

        # append a dummy response
        dialog_context['dialog'].append({
            'text': 'n/a',
            'speaker': 'sys',
        })
        logging.info("******* %s" % (json.dumps(dialog_context)))
        inputs = self.inputter.convert_data_to_inputs(dialog_context)
        features = self.inputter.convert_inputs_to_features(inputs)
        collated_features = self.inputter.convert_features_for_model(features, infer=True)
        collated_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                           collated_features.items()}
        collated_inputs.update(self.generation_kwargs)

        # model generate
        generations = self.model.generate(**collated_inputs)
        output_tokens = generations[0].tolist()
        output_text = self.tokenizer.decode(output_tokens, remove_special_tokens=True)
        output_text = output_text.strip()
        output_text_list = sents_tokenizer.tokenize(output_text)
        output_text_list = [o.capitalize() for o in output_text_list]
        output_text = ' '.join(output_text_list)
        output_text = re.sub('__end__', '', output_text)
        output_text = re.sub('__unk__', '', output_text)
        output_text = re.sub("i'", "I'", output_text)
        output_text = re.sub(' i ', ' I ', output_text)
        output_text = re.sub(' don ; t ', " don't ", output_text)
        logging.info("Response: {}".format(output_text))

        return output_text
