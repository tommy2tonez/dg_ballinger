
class Config:
    huffman_alphabet_sz: int #1,2
    token_dim: int #encoding dimension (768)
    has_padding: bool #inp_token_sz == out_token_sz (break single resposibility here - but for good reason)
    batch_sz: int #training batch pairified (break single resposibility here - but for good reason)
    training_data_path: str #inp_path
    out_path: str #tokenized path 

#out_path only for training_data_extractor (once) - should return data here
def tokenize(config: Config, verbose = True):
    pass  

# given training data set [(inp, out)] - do huffman_encoding + tokenize into 9.58 bits (768/element) 