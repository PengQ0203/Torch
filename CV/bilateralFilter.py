
import  cv2
def bi_demo(image)  :  # 高斯双边滤波
    dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
    cv2.namedWindow('bi_demo' ,0)
    cv2.resizeWindow('bi_demo' ,300 ,400)
    cv2.imshow("bi_demo", dst)

'''
    其中各参数所表达的意义：
    src：原图像；
    d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；
    sigmaColor：颜色空间的标准方差，一般尽可能大；
    sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。'''

def mean_shift_demo(image)  :  # 均值偏移滤波
    dst = cv2.pyrMeanShiftFiltering(src=image, sp=15, sr=20)
    # cv2.namedWindow('mean_shift image', 0)
    # cv2.resizeWindow('mean_shift image', 300, 400)

    return dst



def mean_shift_filter(img, sp=10, sr=30, max_level=50, eps=1e-6):
    row, col = img.shape[:2]
    frame1 = img[int(0.2 * row):,int(col * 0.32):int(col * 0.84)]

    scale_percent = 0.3
    row1, col1 = frame1.shape[:2]
    width = int(row1 * scale_percent)
    height = int(col1 * scale_percent)
    dim = (width, height)
    img_ds = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
    dst = cv2.pyrMeanShiftFiltering(src=img_ds, sp=15, sr=20)
    result = cv2.resize(dst, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)
    img[int(0.2 * row):,int(col * 0.32):int(col * 0.84)] = result
    return img



# 使用均值边缘保留滤波时，可能会导致图像过度模糊
'''其中各参数所表达的意义：
    src：原图像;
    sp：空间窗的半径(The spatial window radius);
    sr：色彩窗的半径(The color window radius)'''

path1 = "../video/202-233/part3big/images/00000045.png"
# src = cv2.imread(path1)
cap = cv2.VideoCapture('../video/202-233/FormatFactoryPart3.mp4')

while (cap.isOpened()):
    # ret返回布尔值
    ret, frame = cap.read()
    dst = mean_shift_filter(frame)
    # row, col = frame.shape[:2]
    # frame1 = frame[int(col * 0.32):int(col*0.84),int(0.2*row):]
    #
    # scale_percent = 0.3
    # row1, col1 = frame.shape[:2]
    # width = int(row1 * scale_percent)
    # height = int(col1 * scale_percent)
    # dim = (width, height)
    # img_ds = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
    # # 展示读取到的视频矩阵
    # # bi_demo(frame)
    # img_mean = mean_shift_demo(img_ds)
    # result = cv2.resize(img_mean, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)
    # # cv2.namedWindow('src', 0)
    # # cv2.resizeWindow('src', 300, 400)
    cv2.imshow('src', frame)
    cv2.imshow("mean_shift image", dst)
    k = cv2.waitKey(2)
    # q键退出
    if k & 0xFF == ord('q'):
        break

# cv2.waitKe


