"""
generate.py
-----------------
"""

# imports 
import torch
from tokenizers import Tokenizer
from model import Transformer, Config

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load checkpoint
    ckpt = torch.load('ckpt.pt', map_location=device, weights_only=False)
    config = Config(**ckpt['config'])
    meta = ckpt['meta']

    model = Transformer(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    tok = Tokenizer.from_file(meta['tokenizer_path'])
    eot_id = meta.get('eot_id')

    while True:
        prompt = input('Enter Prompt: ')
        
        if prompt == '':
            prompt = 'Once upon a time'
        if prompt == 'exit':
            break
        prompt_ids = tok.encode(prompt).ids
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = model.generate(context, max_new_tokens=200, 
                             temperature=0.8, top_k=200 )
        ids = out[0].tolist()

        if eot_id is not None and eot_id in ids[len(prompt_ids):]:
            ids = ids[: ids.index(eot_id, len(prompt_ids))]

        print('\n----story start----')
        print(tok.decode(ids))
        print('----story end----\n')

if __name__ == '__main__':
    main()
    