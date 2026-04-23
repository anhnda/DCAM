import torch
import torch.nn as nn

class FineTunedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedModel, self).__init__()
        
        # ---------------------------------------------------
        # PHẦN 1: TRÍCH XUẤT ĐẶC TRƯNG (Giống hệt AlexNet)
        # ---------------------------------------------------
        self.feature_extractor = nn.Sequential(
            # (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # (1): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            # (2): MaxPool2d(kernel_size=3, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # (3): Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2))
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # (4): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            # (5): MaxPool2d(kernel_size=3, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # (6): Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1))
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # (7): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            
            # (8): Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1))
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # (9): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            
            # (10): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (11): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            # (12): MaxPool2d(kernel_size=3, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # ---------------------------------------------------
        # PHẦN 2: PHÂN LOẠI (Classifier Head)
        # ---------------------------------------------------
        self.classifier = nn.Sequential(
            # (0): AdaptiveAvgPool2d(output_size=(6, 6))
            # Lớp này đảm bảo output của feature_extractor luôn là (N, 256, 6, 6)
            nn.AdaptiveAvgPool2d((6, 6)),
            
            # (1): Flatten(start_dim=1, end_dim=-1)
            # Làm phẳng (N, 256, 6, 6) -> (N, 9216)
            nn.Flatten(),
            
            # (2): Dropout(p=0.5)
            nn.Dropout(p=0.5),
            # (3): Linear(in_features=9216, out_features=4096)
            # 9216 = 256 (channels) * 6 * 6 (từ AdaptiveAvgPool)
            nn.Linear(256 * 6 * 6, 4096),
            # (4): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            
            # (5): Dropout(p=0.5)
            nn.Dropout(p=0.5),
            # (6): Linear(in_features=4096, out_features=4096)
            nn.Linear(4096, 4096),
            # (7): ReLU(inplace=True)
            nn.ReLU(inplace=True),
            
            # (8): Linear(in_features=4096, out_features=10)
            # Lớp output, 10 ở đây là số class (ví dụ: 10 lớp của Imagenette)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # Chạy qua các lớp Conv
        x = self.feature_extractor(x)
        # Chạy qua các lớp Linear (Fully Connected)
        x = self.classifier(x)
        return x
    
