from visdial.decoders.disc import DiscriminativeDecoder
from visdial.decoders.gen import GenerativeDecoder

def Decoder(hparams, *args):
	name_dec_map = {
		"disc": DiscriminativeDecoder,
		"gen": GenerativeDecoder,
	}
	return name_dec_map[hparams.decoder](hparams, *args)
