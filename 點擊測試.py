import matplotlib.pyplot as plt
import cv2
# 定義滑鼠點擊事件處理函數
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        print(f"滑鼠座標 (h, w): ({event.ydata:.0f}, {event.xdata:.0f})")

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
image = cv2.imread(r"C:\Users\USER\Pictures\images.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將 BGR 格式轉換為 RGB 格式
image = preprocessing(image)

# 顯示圖片
fig, ax = plt.subplots()
ax.imshow(image)

# 綁定滑鼠點擊事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()