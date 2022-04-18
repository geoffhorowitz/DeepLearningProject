import torch.nn as nn
import torchvision.models as models


class Im2Recipe(nn.Module):

    def __init__(self):
        super(Im2Recipe, self).__init__()

        # Image model
        cnn = models.resnet50(pretrained=True)
        # 2048 is featureDim of input of last fc
        # hard-coding 1024 as embedding dim but can change later
        cnn.fc = nn.Linear(2048, 1024)
        self.image_model = cnn

        # TODO: Initialize recipe models

    def forward(self, x):
        out_image = self.image_model(x)
        # TODO: Add recipe outputs
        return out_image, None
