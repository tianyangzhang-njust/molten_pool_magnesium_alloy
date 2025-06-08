import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
model = torch.load('checkpoint/best.pth')
model = model.eval().to(device)
from torchcam.methods import GradCAMpp
cam_extractor = GradCAMpp(model, target_layer=model.layer4)
from torchvision import transforms
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
img_path = 'WFS9.5.jpg'
img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
pred_logits = model(input_tensor)
pred_id = torch.topk(pred_logits, 1)[1].detach().cpu().numpy().squeeze().item()
activation_map = cam_extractor(pred_id, pred_logits)
activation_map = activation_map[0][0].detach().cpu().numpy()
plt.imshow(activation_map)
plt.show()
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.6)
plt.show(result)
plt.imshow(result)


