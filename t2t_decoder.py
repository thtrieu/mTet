"""Decode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_decoder
from tensor2tensor.models import transformer
import problems
import tensorflow as tf
import decoding
from tensor2tensor.utils import registry

flags = tf.flags
FLAGS = flags.FLAGS


@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  return hparams


@registry.register_hparams
def transformer_tall_18_18():
  hparams = transformer_tall9()
  hparams.num_encoder_layers = 18
  hparams.num_decoder_layers = 18
  return hparams


@registry.register_model
class Transformerextratokentodecoderv2(transformer.Transformer):

  def encode(self, *args, **kwargs):
    encoder_output, encoder_decoder_attention_bias = super(
        Transformerextratokentodecoderv2, self).encode(*args, **kwargs)
    hparams = self._hparams
    
    batch_size = encoder_output.shape[0]
    hidden_dim = int(encoder_output.shape[-1])
    
    num_extras = hparams.extra_tokens
    extra_tokens = tf.get_variable(
        'extra_tokens', [1, num_extras, hidden_dim],
        initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5))
    
    extra_tokens = tf.tile(
        extra_tokens, [batch_size, 1, 1], axis=0)  # [batch, num_extras, hidden_dim]
    encoder_output = tf.concat(
        [extra_tokens, encoder_output], axis=1)  # [batch, num_extras+len, hidden_dim]

    encoder_decoder_attention_bias = tf.pad(
        encoder_decoder_attention_bias,
        [[0, 0], [num_extras, 0]],
        constant_values=1.0,
    )
    return encoder_output, encoder_decoder_attention_bias


@registry.register_hparams
def transformer_tall9_extra_tokens():
  hparams = transformer_tall9()
  hparams.add_hparam("extra_tokens", FLAGS.extra_tokens)
  return hparams


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(t2t_decoder.main)
  decoding.t2t_decoder(
      FLAGS.problem, 
      FLAGS.data_dir, 
      FLAGS.decode_from_file, 
      FLAGS.decode_to_file,
      FLAGS.checkpoint_path or FLAGS.output_dir)
