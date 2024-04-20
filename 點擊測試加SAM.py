import matplotlib.pyplot as plt
import cv2
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits # type: ignore
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile

# 定義全局變數來存儲座標值
clicked_h = None
clicked_w = None 

# 定義滑鼠點擊事件處理函數
def onclick(event):
    global clicked_h , clicked_w
    if event.xdata is not None and event.ydata is not None:
        print(f"滑鼠座標 (h, w): ({event.ydata:.0f}, {event.xdata:.0f})")
        h = round(event.ydata)
        w = round(event.xdata)
        clicked_h=h
        clicked_w=w

def preprocessing(image):                                               # 前處理
    #================================================ 將圖象變成正方形 ==================================================
    h, w, _ = image.shape
    padding_size = abs(w - h)

    # 計算預padding的上下左右邊框大小
    top = padding_size // 2
    bottom = padding_size - top
    left = padding_size // 2
    right = padding_size - left

    # padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 确定正方形邊長
    square_size = max(h, w)

    # 計算預裁切的區域
    start_x = (padded_image.shape[1] - square_size) // 2
    start_y = (padded_image.shape[0] - square_size) // 2

    # 裁切
    transformed_image = padded_image[start_y:start_y+square_size, start_x:start_x+square_size, :]
    
    return transformed_image


# 讀取圖片
image_file = r"C:\Users\USER\Pictures\images.jpg"
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將 BGR 格式轉換為 RGB 格式
image = preprocessing(image)

# 顯示圖片
fig, ax = plt.subplots()
ax.imshow(image)

# 綁定滑鼠點擊事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
###################################
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

input_points = torch.tensor([[[[clicked_w , clicked_h]]]])
input_labels = torch.tensor([[[1]]])   
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_points(input_points, input_labels, plt.gca())
plt.axis('on')
plt.show()  
###################################
models = {}
sample_image_np = np.array(Image.open(image_file))
print("Shape of original_image:", sample_image_np.shape)
sample_image_tensor = transforms.ToTensor()(sample_image_np)
# Build the EfficientSAM-Ti model.
models['efficientsam_ti'] = build_efficient_sam_vits()
input_points = torch.tensor([[[[clicked_w , clicked_h]]]])
input_labels = torch.tensor([[[1]]])
for model_name, model in models.items():
    print('Running inference using ', model_name)
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
    print("Shape of sample_image_tensor:", sample_image_tensor.shape)
    print("Shape of mask:", mask.shape)
    print("Shape of mask:", mask)
    print("Type of masked_image_np:", type(masked_image_np))
    plt.imshow(masked_image_np)
    plt.axis('off')  # 不顯示座標軸
    plt.show()