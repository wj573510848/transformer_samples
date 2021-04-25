from transformers import TFGPT2LMHeadModel
from transformers.models.gpt2.modeling_tf_gpt2 import GPT2_INPUTS_DOCSTRING, add_start_docstrings_to_model_forward, add_code_sample_docstrings, _TOKENIZER_FOR_DOC, TFCausalLMOutputWithPast, _CONFIG_FOR_DOC, input_processing

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
import tensorflow as tf
from transformers.modeling_tf_utils import shape_list

class ChidGPT2Model(TFGPT2LMHeadModel):
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            training=False,
            **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            past=inputs["past"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        hidden_states = transformer_outputs[0]
        logits = self.transformer.wte(hidden_states, mode="linear")

        losses = None
        if labels is not None:
            mask = tf.not_equal(labels, -100)
            
            labels = tf.cast(labels,tf.int32) * tf.cast(mask,tf.int32) 
            raw_loss = loss_fun(labels,logits)

            mask = tf.cast(mask,tf.float32)
            losses = tf.cast(raw_loss,tf.float32) * mask
            losses = tf.reduce_sum(losses,axis=-1) / tf.reduce_sum(mask,axis=-1)
        
        return losses

def loss_fun(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss_fn(labels, logits)



