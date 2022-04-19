"""
@article{marin2019learning,
  title = {Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images},
  author = {Marin, Javier and Biswas, Aritro and Ofli, Ferda and Hynes, Nicholas and
  Salvador, Amaia and Aytar, Yusuf and Weber, Ingmar and Torralba, Antonio},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  year = {2019}
}

@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for Cooking Recipes and Food Images},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
-----do not edit anything above this line---
"""

import numpy as np

'''
reference: https://github.com/torralba-lab/im2recipe-Pytorch/blob/master/scripts/

used as a helper function to translate Recipe 1M dataset format to vectors used in RNN
'''

def load_vocab(p, offset=1):
    with open(p) as f_vocab:
        vocab = {w.rstrip(): i+offset for i, w in enumerate(f_vocab)}
    return vocab


def detect_ingrs(recipe, vocab):
    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients'] if ingr['text']]
    except:
        ingr_names = []
        print("Could not load ingredients! Moving on...")

    detected = set()
    for name in ingr_names:
        name = name.replace(' ','_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)
        '''
        name_words = name.lower().split(' ')
        for i in xrange(len(name_words)):
            name_ind = vocab.get('_'.join(name_words[i:]))
            if name_ind:
                detected.add(name_ind)
                break
        '''

    return list(detected) + [vocab['</i>']]
