import numpy as np
import sys
import cv2
import os
import json
import glob
import time

import DobotDllType as dType


from model_v3 import mobilenet_v3_large
from MvCameraControl_class import *

font = cv2.FONT_HERSHEY_SIMPLEX

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

api = dType.load()

state = dType.ConnectDobot(api, "COM6", 115200)[0]
print("Connect status:",CON_STR[state])


lower_white = np.array([40, 0, 170])       # 红色阈值下界
higher_white = np.array([110, 25, 255])     # 红色阈值上界

# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
json_file = open(json_path, "r")
class_indict = json.load(json_file)
# create model
model = mobilenet_v3_large(num_classes = 5)
weights_path = './save_weights/resMobileNetV3.ckpt'
assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
model.load_weights(weights_path)


cab_long = 350 #340-53
car_long = 350
pear_long = 350
pine_long = 350
straw_long = 350

def Enum_device(tlayerType, deviceList):
    """
    ch:枚举设备 | en:Enum device
    nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    """
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
def enable_device(nConnectionNum):
    """
    设备使能
    :param nConnectionNum: 设备编号
    :return: 相机, 图像缓存区, 图像数据大小
    """
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # 从这开始，获取图片数据
    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    # MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
    # strKey [IN] 属性键值，如获取宽度信息则为"Width"
    # pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
    # 得到图片尺寸，这一句很关键
    # payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()

    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    #  返回获取图像缓存区。
    data_buf = (c_ubyte * nPayloadSize)()
    #  date_buf前面的转化不用，不然报错，因为转了是浮点型
    return cam, data_buf, nPayloadSize
def get_image(data_buf, nPayloadSize):
    """
    获取图像
    :param data_buf:
    :param nPayloadSize:
    :return: 图像
    """
    # 输出帧的信息
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    # void *memset(void *s, int ch, size_t n);
    # 函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s
    # memset:作用是在一段内存块中填充某个给定的值，它是对较大的结构体或数组进行清零操作的一种最快方法
    # byref(n)返回的相当于C的指针右值&n，本身没有被分配空间
    # 此处相当于将帧信息全部清空了
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    # 采用超时机制获取一帧图片，SDK内部等待直到有数据时返回，成功返回0
    ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
    # if ret == 0:
    #     print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
    #         stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
    # else:
    #     print("no data[0x%x]" % ret)

    image = np.asarray(data_buf)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
    image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))  # 根据自己分辨率进行转化
    return image
def close_device(cam, data_buf):
    """
    关闭设备
    :param cam:
    :param data_buf:
    """
    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf

def predict(img):
    start_time = time.time()#获取当前时间

    im_height = 224
    im_width = 224
    num_classes = 5

    img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#转换色制 不然颜色不对
    img = cv2.resize(img, dsize=(im_height, im_width))

    scale_img = img.copy()#复制一份来显示图像
    scale_img = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)  # 默认是BRG，要转化成RGB，颜色才正常

    # scaling pixel value to (-1,1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))



    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "{}  {}%".format(class_indict[str(predict_class)],
                                                (round(result[predict_class],2)*100))

    cv2.putText(scale_img, print_res, (10, 25), font, 0.7, (0, 0, 255), 2)
    cv2.imshow("scale_img",scale_img)
    print(result[predict_class])
    predict_time = time.time() - start_time #计算推理模型用的时间
    return (class_indict[str(predict_class)])


i = [0]
flag1 = 0
flag2 = 0
if (state == dType.DobotConnect.DobotConnect_NoError):
    dType.SetQueuedCmdClear(api)# 初始化清空机械臂的指令
    dType.SetQueuedCmdStartExec(api)# 开始执行队列指令
    dType.SetEndEffectorParams(api, 59.7, 0, 0, 1) # 设置机械臂末端为夹爪
    dType.SetEndEffectorGripper(api, 1, 0, isQueued=1)#张开
    dType.SetInfraredSensor(api, 1, 1, version=0)  # 使能光电传感器

    

    
    dType.SetPTPCmd(api, 1,117,278,90,0, 1)
    dType.SetEMotorEx(api, 0, 1, 2000, isQueued=1)  #10-85  8500 传送带动作 7500pulse/s
    print("init......s")



# 获得设备信息
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

# ch: 枚举设备 | en:Enum device
# nTLayerType[IN] 枚举传输层 ，pstDevList[OUT] 设备列表
Enum_device(tlayerType, deviceList)

# 获取相机和图像数据缓存区
cam, data_buf, nPayloadSize = enable_device(0)  # 选择第一个设备
while True:
    image = get_image(data_buf, nPayloadSize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 默认是BRG，要转化成RGB，颜色才正常
    image = cv2.resize(image, dsize=(648, 486))

    image = image[0:486,90:520] #剪裁图像

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 50), (179, 50, 90))# 根据像素的范围进行过滤，把符合像素范围的保留，不符合的赋值0或者255 # 根据hsv颜色表找出最大值和最小值
    mask = cv2.bitwise_not(mask)    # 只在mask区域做与运算
    cnts1, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #找边缘
  

    W_cy = 486


    for cnt in cnts1:        #                    b g r
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点  x y为正方框的左上角点
        if w*h <20000 :
            continue
        rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        box = cv2.boxPoints(rect) #  获取最小外接矩形的4个顶点坐标
        box = np.int0(box)#取整
        image = cv2.drawContours(image, [box], 0, (255, 0, 0), 2)#画旋转方框(贴合边缘)

        if(int(rect[0][1]) < W_cy ) : #在相机坐标中比较这个颜色y方向的位置与上一个颜色y坐标位置，较小说明在这个颜色传送带的前边
            W_x = int(box[1][0] + (box[2][0] - box[1][0])/2)    #获取一条边的中心点x坐标  发给机械臂用
            W_y = int(box[1][1] + (box[2][1] - box[1][1])/2)
            W_rot = 90 - int(rect[2])

            image = cv2.line(image,(W_x+15,W_y),(W_x-15,W_y),(0, 255, 0),3)#画中心十字
            image = cv2.line(image,(W_x,W_y+15),(W_x,W_y-15),(0, 255, 0),3) 
            # cv2.line(image,(box[1]),(box[2]),(0,255,0),2)
            # cv2.circle(image,(W_x,W_y),3, (0,0,255), -1)

            cx = int(rect[0][0])#获取中心点x坐标  旋转分割图像用
            cy = int(rect[0][1])
            rotation = rect[2]
            width = int(rect[1][0])
            height = int(rect[1][1])

            robot_x = int(65 - (W_y / 4.3))#计算机械臂的坐标
            robot_y = int(224 + (520 - W_x)/4.3)
            rot = W_rot

            destination_z = 5


    # if(flag2 == 0): #机械臂每抓完一次才加
    #     flag2 = 1
    #     W_long = W_long - 53 #每次减一个盒子的宽度
    #     destination_x = W_long
    
    i = dType.GetInfraredSensor(api, 1)#获取光电传感器状态
    if i == [1]  :#有东西
        dType.SetEMotorEx(api, 0, 0, 0, isQueued=1)#传送带停止
        if(flag1 == 0):
            flag1 = 1
            M = cv2.getRotationMatrix2D((cx,cy), rotation, 1)#设置中心点 旋转角度 比例  作为参数M
            img_rot = cv2.warpAffine(image, M, (648, 486))#使用参数M进行旋转
            x1 = int(cx-width/2)
            x2 = int(cx+width/2)
            y1 = int(cy-height/2)
            y2 = int(cy+height/2)
            img_rot = img_rot[y1:y2,x1:x2]#剪裁旋转后的图像
            result = predict(img_rot)
            if(result == "cabbage"):
                destination_y = 110    #要放到哪个位置的机械臂y坐标
                cab_long = cab_long - 53 #每次减一个盒子的宽度
                destination_x = cab_long
            elif(result == "carrot"):
                destination_y = 55    #要放到哪个位置的机械臂y坐标
                car_long = car_long - 53 #每次减一个盒子的宽度
                destination_x = car_long
            elif(result == "pear"):
                destination_y = 0    #要放到哪个位置的机械臂y坐标
                pear_long = pear_long - 53 #每次减一个盒子的宽度
                destination_x = pear_long
            elif(result == "pineapple"):
                destination_y = -55    #要放到哪个位置的机械臂y坐标
                pine_long = pine_long - 53 #每次减一个盒子的宽度
                destination_x = pine_long
            elif(result == "strawberry"):
                destination_y = -110    #要放到哪个位置的机械臂y坐标
                straw_long = straw_long - 53 #每次减一个盒子的宽度
                destination_x = straw_long



        dType.SetPTPCmd(api, 1,robot_x-10,robot_y-25,100,rot, 1)#到方块上方
        dType.dSleep(1000)
        dType.SetPTPCmd(api, 1,robot_x-10,robot_y-25,66,rot, 1)#下去
        dType.dSleep(1000)
        dType.SetEndEffectorGripper(api, 1, 1, isQueued=1)#闭合
        dType.dSleep(1000)
        dType.SetPTPCmd(api, 1,robot_x-10,robot_y-25,75,rot, 1)#抓到后上去一点  避免回去的时候撞到传送带
        dType.dSleep(500)
        dType.SetPTPCmd(api, 1,destination_x,destination_y,75,-90, 1)#到分类仓库头顶
        dType.dSleep(1000)
        dType.SetPTPCmd(api, 1,destination_x,destination_y,destination_z,-90, 1)#下去
        dType.dSleep(1000)
        dType.SetEndEffectorGripper(api, 1, 0, isQueued=1)#张开
        dType.dSleep(1000)
        dType.SetPTPCmd(api, 1,destination_x,destination_y,40,-90, 1)#上来
        dType.dSleep(1000)
        dType.SetPTPCmd(api, 1,117,278,70,0, 1)#准备抓东西的位置
        dType.SetEMotorEx(api, 0, 1, 2000, isQueued=1)
        
    else:
        flag1 = 0 #东西被夹走了 下一次预测才能进行

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', image)

    key = cv2.waitKey(10)    
    # if int(key) == 115:#按下s键暂停图像
    #     print("stop!")

    #     M = cv2.getRotationMatrix2D((cx,cy), rot, 1)#设置中心点 旋转角度 比例  作为参数M
    #     img_rot = cv2.warpAffine(image, M, (648, 486))#使用参数M进行旋转
    #     x1 = int(cx-width/2)
    #     x2 = int(cx+width/2)
    #     y1 = int(cy-height/2)
    #     y2 = int(cy+height/2)
    #     img_rot = img_rot[y1:y2,x1:x2]#剪裁旋转后的图像

    #     predict(img_rot)
        

        
    #     while True:
    #         key = cv2.waitKey(10)
    #         if int(key) == 115:#再按一次s键 继续传输图像
    #             break
    #         if cv2.waitKey(1) & 0xFF == ord('q'):#按下q键退出程序
    #             cv2.destroyAllWindows()
    #             break
    # if cv2.waitKey(1) & 0xFF == ord('q'):#按下q键退出程序
    #     cv2.destroyAllWindows()
    #     break

# 关闭设备
close_device(cam, data_buf)