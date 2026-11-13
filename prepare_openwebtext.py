import argparse
import os
import time
from pathlib import Path

import numpy as np

def tokenize(example):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

def main():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--out', default='data/openwebtext')
    p.add_argument('--num-proc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    out = Path(args.out)
    if (out / 'train.bin').exists() and (out / 'val.bin').exists():
        print(f'{out} already holds train.bin and val.bin')
        return
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    dataset = load_dataset('Skylion007/openwebtext', num_proc=args.num_proc)
    split = dataset['train'].train_test_split(test_size=0.005, seed=42, shuffle=True)
    split['val'] = split.pop('test')
    tokenized = split.map(tokenize, remove_columns=['text'], num_proc=args.num_proc, desc='tokenizing')
    for name, dset in tokenized.items():
        total = int(np.sum(dset['len'], dtype=np.int64))
        arr = np.memmap(str(out / f'{name}.bin'), dtype=np.uint16, mode='w+', shape=(total,))
        idx, t0 = 0, time.time()
        for i, ex in enumerate(dset.with_format('numpy')):
            ids = ex['ids'].astype(np.uint16)
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
            if (i + 1) % 500_000 == 0:
                print(f'  {name}: {i + 1:,}/{len(dset):,} documents ({(time.time() - t0) / 60:.1f} min)')
        assert idx == total
        arr.flush()
        print(f'{name}.bin: {total:,} tokens')

if __name__ == '__main__':
    main()
