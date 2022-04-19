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

import word2vec
import sys
import os

'''
Usage: python get_vocab.py /path/to/vocab.bin

reference: https://github.com/torralba-lab/im2recipe-Pytorch/blob/master/scripts/

used as a helper function to translate Recipe 1M dataset format to vectors used in RNN
'''
w2v_file = sys.argv[1]
model = word2vec.load(w2v_file)

vocab =  model.vocab

print("Writing to %s..." % os.path.join(os.path.dirname(w2v_file),'vocab.txt'))
f = open(os.path.join(os.path.dirname(w2v_file),'vocab.txt'),'w')
f.write("\n".join(vocab))
f.close()
