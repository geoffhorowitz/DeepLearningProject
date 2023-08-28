# DeepLearningProject

Team Name: Alphabet Soup

Project Title: Recognizing image-recipe pairs using cross-modal learning

# Final Paper
See the final paper including problem definition, process outline, and results [here](Report/Project_Report___CS_7643.pdf)

# Data
The database for this project is the Recipe 1M dataset

For convenience, the used data can be found [here](Database/DataDrive.txt)

# Environment
the necessary environment files can be found [here](environment.yaml)  
using anaconda, this can be imported with following command:
```
conda env create -f environment.yaml
```

# LSTM setup instructions
## generating vocab
python utils/unpack_vocab.py /path/to/vocab.bin
