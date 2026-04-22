import torch
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Model:
    def __init__(self, pretrained=True, num_classes=10):
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        last_layer_num_features = self.model.fc.in_features
        self.classifier = torch.nn.Linear(last_layer_num_features, num_classes)
        self.model.fc = self.classifier

    def forward(self, x):
        output = self.model(x)
        return output

    def predict(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
    

if __name__ == "__main__": 
    model = ResNet18Model(pretrained=True, num_classes=10)
    print(model)

    sample_input = torch.randn(1, 3, 224, 224)
    sample_output = model.forward(sample_input)
    print(sample_output)