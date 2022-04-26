import torch
import torch.nn as nn
import torchvision.models as models

from lstm.lstm_model import RecipeModel
from lstm.lstm_model import IngredModel

class Im2Recipe(nn.Module):

    def __init__(self, args):
        super(Im2Recipe, self).__init__()

        # Image model
        cnn = models.resnet50(pretrained=True)
        # freeze the layers
        cnn_children = []
        for child in cnn.children():
            for param in child.parameters():
                param.requires_grad = False
            cnn_children.append(child)
        # remove final layer from frozen cnn
        self.frozen_image_model = nn.Sequential(*cnn_children[:-1])
        # 2048 is featureDim of input of last fc
        self.unfrozen_image_layer = nn.Sequential(
            nn.Linear(2048, args.embed_dim),
            nn.Tanh(),
            nn.LayerNorm(args.embed_dim)
        )
        # self.relu = nn.ReLU()
        # self.class_linear = nn.Linear(args.embed_dim, args.num_classes)

        self.recipe_linear = nn.Linear(args.ingredient_embedding_dim*2 + args.recipe_embedding_dim, args.embed_dim)
        self.recipe_tanh = nn.Tanh()
        self.recipe_norm = nn.LayerNorm(args.embed_dim)
        self.ingred_model = IngredModel(args)
        self.recipe_model = RecipeModel(args)
        if args.semantic_reg:
            self.semantic_layer = nn.Linear(args.embed_dim, args.num_classes)
        else:
            self.semantic_layer = None

    def forward(self, x):
        out_image = self.frozen_image_model(x[0])
        out_image = self.unfrozen_image_layer(out_image.reshape((out_image.shape[0], out_image.shape[1])))

        ingred_output = self.ingred_model(x)
        recipe_output = self.recipe_model(x)
        # print(ingred_output.size())
        # print(recipe_output.size())
        out_recipe = torch.cat((recipe_output, ingred_output), 1)
        out_recipe = self.recipe_tanh(self.recipe_linear(out_recipe))
        out_recipe = self.recipe_norm(out_recipe)
        if self.semantic_layer is not None:
            out_image_reg = self.semantic_layer(out_image)
            out_recipe_reg = self.semantic_layer(out_recipe)
        else:
            out_image_reg = out_recipe_reg = None

        return out_image, out_recipe, out_image_reg, out_recipe_reg
