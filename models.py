import torch
import torch.nn as nn
import torchvision.models as models

#from lstm.lstm_model import RecipeModel_ref as RecipeModel
#from lstm.lstm_model import IngredModel_ref as IngredModel
from lstm.lstm_model import RecipeModel
from lstm.lstm_model import IngredModel

class Im2Recipe(nn.Module):

    def __init__(self, args):
        super(Im2Recipe, self).__init__()

        # Image model
        cnn = models.resnet50(pretrained=True)
        # 2048 is featureDim of input of last fc
        # hard-coding 1024 as embedding dim but can change later
        cnn.fc = nn.Linear(2048, args.embed_dim)
        self.image_model = cnn
        self.relu = nn.ReLU()
        self.class_linear = nn.Linear(args.embed_dim, args.num_classes)

        # TODO: Initialize recipe models
        self.recipe_linear = nn.Linear(args.ingredient_embedding_dim*2 + args.recipe_embedding_dim, args.embed_dim)
        self.recipe_tanh = nn.Tanh()
        self.recipe_norm = nn.LayerNorm(args.embed_dim)
        self.ingred_model = IngredModel(args)
        self.recipe_model = RecipeModel(args)

    def forward(self, x):
        out_image = self.class_linear(self.relu(self.image_model(x[0])))

        # TODO: Add recipe outputs
        ingred_output = self.ingred_model(x)
        recipe_output = self.recipe_model(x)
        print(ingred_output.size())
        print(recipe_output.size())
        out_recipe = torch.cat((recipe_output, ingred_output), 2)
        out_recipe = self.recipe_tanh(self.recipe_linear(out_recipe))
        out_recipe = self.recipe_norm(out_recipe)

        return out_image, out_recipe
