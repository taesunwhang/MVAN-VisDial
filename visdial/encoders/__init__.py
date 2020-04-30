from visdial.encoders.mvan.mvan import MVANEncoder

def Encoder(hparams, *args):
  name_enc_map = {
    "mvan": MVANEncoder,  # Ours
  }
  return name_enc_map[hparams.encoder](hparams, *args)