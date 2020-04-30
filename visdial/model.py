from torch import nn

class EncoderDecoderModel(nn.Module):
  """Convenience wrapper module, wrapping Encoder and Decoder modules.

      Parameters
      ----------
      encoder: nn.Module
      decoder: nn.Module
  """

  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, batch):
    encoder_output = self.encoder(batch)
    decoder_output = self.decoder(encoder_output, batch)
    return decoder_output

class MultiEncoderDecoderModel(nn.Module):
  """Convenience wrapper module, wrapping Encoder and Decoder modules.

      Parameters
      ----------
      encoder: nn.Module
      decoder: nn.Module
  """

  def __init__(self, encoder, disc_decoder=None, gen_decoder=None):
    super().__init__()
    self.encoder = encoder
    self.disc_decoder = disc_decoder
    self.gen_decoder = gen_decoder

  def forward(self, batch):
    disc_decoder_output, gen_decoder_output = None, None
    encoder_output = self.encoder(batch)
    if self.disc_decoder is not None:
      disc_decoder_output = self.disc_decoder(encoder_output, batch)
    if self.gen_decoder is not None:
      gen_decoder_output = self.gen_decoder(encoder_output, batch)

    return disc_decoder_output, gen_decoder_output