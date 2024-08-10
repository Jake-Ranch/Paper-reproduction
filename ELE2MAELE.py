import torch
import math
import numpy as np
import pygame
import torch.nn
import os
import pandas as pd
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from scipy.stats import binom
from math import atan,sin,cos
import re
from IPython import display
import random


caridlist=[]
min_RSU_jing = 1800
min_RSU_wei = 1800
nothing=-20
rount=0
action_dim=0

preLSTM=1
pretransformer=0
pretransformerLSTM=0


# prepicpath='E:\BaiduNetdiskDownload\SCI\prepic\\'
drawpklpath='E:\BaiduNetdiskDownload\SCI\drawpkl9\\'
# prepicpath='E:\BaiduNetdiskDownload\SCI\T_L_C\\'
prepicpath = 'E:\BaiduNetdiskDownload\SCI\LSTMpic\\'

def match(text):
    matches = re.findall("'(.*?)'", text)
    mlist=[]
    for match in matches:
        mlist.append(match)
        # print(match)
    if mlist!=[]:
        return 1
    else:
        return 0

def file_exists(file_path):
    return os.path.exists(file_path)

def ball_dis(jingduA, weiduA, jingduB, weiduB):  # 输入AB点的经纬度，输出球面距离（输入是度数制，非弧度）
    a = (math.sin(Pi/180*(weiduA / 2 - weiduB / 2))) ** 2
    b = math.cos(weiduA * Pi / 180) * math.cos(weiduB * Pi / 180) * (
        math.sin((jingduA / 2 - jingduB / 2) * Pi / 180)) ** 2
    L = 2 * R * math.asin((a + b) ** 0.5)
    return L


def save_agent(agent, filename='demo.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)

def load_agent(filename='demo.pkl'):
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    return agent


namenum='1'
shapping=0
masking=1

pre=1
perfectpre=0


if preLSTM:
    prepicpath = 'E:\BaiduNetdiskDownload\SCI\LSTMpic\\'
    truth=load_agent(prepicpath+'LSTMtruth30B.pkl')
    prelist=load_agent(prepicpath+'LSTMprelist30B.pkl')
elif pretransformer:
    prepicpath='E:\BaiduNetdiskDownload\SCI\Tranpic\\'
    truth=load_agent(prepicpath+'transformertruth30B.pkl')
    prelist=load_agent(prepicpath+'transformerprelist30B.pkl')
elif pretransformerLSTM:
    prepicpath='E:\BaiduNetdiskDownload\SCI\T_L_C\\'
    truth=load_agent(prepicpath+'transformer-LSTMtruth30C.pkl')
    prelist=load_agent(prepicpath+'transformer-LSTMprelist30C.pkl')
else:
    prepicpath='E:\BaiduNetdiskDownload\SCI\T_L_B_xThL\\'
    truth=load_agent(prepicpath+'transformer-LSTMtruth29B.pkl')
    prelist=load_agent(prepicpath+'transformer-LSTMprelist29B.pkl')


# elelist = ['', 'nopre', 'perfectpre']
elelist = ['', 'nopre', 'perfectpre','nomask','noUAV','noSATCOM','60','40']
# elelist = ['', 'nopre','nomask','noUAV','noSATCOM','60','40']
# elelist = ['']
# elelist = [ 'nomask','']


UAV_speed=0.001#每个time步移动的距离
UAV_range=1500#无人机服务范围
RSU_range=1200#RSU服务范围
computer = 120#150 #RSU计算能力，计算完才能发送，取值比任务大，但比buffer小，否则buffer没有存在的必要了
maxbuffer = 300#200 #RSU缓存区(最大RSU信道带宽(Mbps)
# mission_size = 30  # *1024**2*8  #10MB 每辆车的任务大小（Bit）
mission_size = 40  # *1024**2*8  #10MB 每辆车的任务大小（Bit）

resetpre=1

# RSUlist=['','noUAV','noSATCOM']
RSUi=0
prei=''

ranlist=[1026315,9145140]#1145140]#1919810]#1018919
randomseed=ranlist[1]#1919810#114514#sum(caridset)#设置随机种子'-'+ str(randomseed)+'-'+
toomuchRSU=0.7#load太大
toomanyRSU=0.1#0.2 #随机保留30%的RSU
# toomanyRSU=0.2#0.2 #随机保留30%的RSU

# 预设值
movelabel=300
starty=200
texty=5
R_SATCOM=16000/1024#822.5    (Mbit/s   Mbps)
SATcomputer=300#卫星计算能力
fly_high=50#UAV飞行高度
len_carfuture = 1  # 预测未来的车辆轨迹点数




w_E = 0.48  # 负载
w_R = 0.5  # 传输时延
w_J = 0.02
w_D = 0.5  # 距离

w_EE = 0.5#0.5  # 负载
w_RR = 0.5  # 传输时延
w_JJ = 1  #一次迁移花费多少时延（可认为在两个RSU之间上传一个完整的mission-size）

plotright=0

if RSUi==1:
    len_UAV = 0  # 无人机数量
else:
    len_UAV = 10  # 5#无人机数量



# caridset=[1,9,5,2,0]
if namenum=='1':
    caridset=[1,9,5,2,77]#可用1
elif namenum=='2':
    caridset=[17,22,46,12,10]#2
elif namenum=='3':
    caridset=[68,31,25,58,11]#3
elif namenum=='4':
    caridset=[34,3,4]#少车

if namenum=='1':
    namemap=(800,700)#1
elif namenum=='2':
    namemap=(650, 400)#2
elif namenum=='3':
    namemap=(750, 600)#3

if namenum=='1':
    namelocal=[-185,90]#1
elif namenum=='2':
    namelocal=[-35,70]#2
elif namenum=='3':
    namelocal=[-60,10]#3


mappo_num = len(caridset)


#MASAC参数
# max_action = 4.29#73-1e-3#env.action_space.high[0]
# max_action = 5#73-1e-3#env.action_space.high[0]
max_action = 7 #73-1e-3#env.action_space.high[0]
min_action = 0.01#env.action_space.low[0]
RENDER = False
# EP_MAX = 800
# EP_LEN = 500
# GAMMA = 0.9
# q_lr = 3e-4
# value_lr = 3e-4
# policy_lr = 1e-4
# tau = 1e-3
GAMMA = 0.9#0.9
q_lr = 2e-3
value_lr = 2e-3
# policy_lr = 1e-3
# policy_lr = 1.5e-3
policy_lr = 1.5e-4
tau = 1e-2
BATCH = 128
# MemoryCapacity = 32768
MemoryCapacity = 100#100个epoch之后才能开始训练
Switch = 0
saveone=0

picsize=20
showsize=30

RSU_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\基站.png')#.convert()
car_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\汽车.png')#.convert()
maincar_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\主汽车.png')#.convert()
road_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\道路.png')#.convert()
UAV_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\无人机.png')#.convert()
UAVcan_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\无人机3.png')#.convert()
UAVwarn_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\无人机警告.png')#.convert()
star_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\卫星.png')#.convert()
warn_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\警告.png')#.convert()
can_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\基站3.png')#.convert()
connect_RSU_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\已连接RSU.png')#.convert()
connect_SATCOM_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\已连接卫星.png')#.convert()
connect_UAV_image = pygame.image.load('E:\BaiduNetdiskDownload\SCI\已连接UAV.png')#.convert()
background = pygame.image.load('E:\BaiduNetdiskDownload\SCI\地图'+namenum+'.png')#.convert()

RSU_image_s=pygame.transform.scale(RSU_image, (showsize, showsize))
car_image_s=pygame.transform.scale(car_image, (showsize, showsize))
maincar_image_s=pygame.transform.scale(maincar_image, (showsize, showsize))
road_image_s=pygame.transform.scale(road_image, (showsize, showsize))
UAV_image_s=pygame.transform.scale(UAV_image, (showsize, showsize))
UAVcan_image_s=pygame.transform.scale(UAVcan_image, (showsize, showsize))
UAVwarn_image_s=pygame.transform.scale(UAVwarn_image, (showsize, showsize))
star_image_s=pygame.transform.scale(star_image, (showsize, showsize))
warn_image_s=pygame.transform.scale(warn_image, (showsize, showsize))
can_image_s=pygame.transform.scale(can_image, (showsize, showsize))
connect_RSU_image_s=pygame.transform.scale(connect_RSU_image, (showsize, showsize))
connect_SATCOM_image_s=pygame.transform.scale(connect_SATCOM_image, (showsize, showsize))
connect_UAV_image_s=pygame.transform.scale(connect_UAV_image, (showsize, showsize))

RSU_image=pygame.transform.scale(RSU_image, (picsize, picsize))
car_image=pygame.transform.scale(car_image, (picsize, picsize))
maincar_image=pygame.transform.scale(maincar_image, (picsize, picsize))
road_image=pygame.transform.scale(road_image, (picsize, picsize))
UAV_image=pygame.transform.scale(UAV_image, (picsize, picsize))
UAVcan_image=pygame.transform.scale(UAVcan_image, (picsize, picsize))
UAVwarn_image=pygame.transform.scale(UAVwarn_image, (picsize, picsize))
star_image=pygame.transform.scale(star_image, (picsize, picsize))
warn_image=pygame.transform.scale(warn_image, (picsize, picsize))
can_image=pygame.transform.scale(can_image, (picsize, picsize))
connect_RSU_image=pygame.transform.scale(connect_RSU_image, (picsize, picsize))
connect_SATCOM_image=pygame.transform.scale(connect_SATCOM_image, (picsize, picsize))
connect_UAV_image=pygame.transform.scale(connect_UAV_image, (picsize, picsize))
background=pygame.transform.scale(background, namemap)


# screen = pygame.display.set_mode((500, 500))
# pygame.init()
# screen.fill((255,255,255))
# screen.blit(road_image, (300, 40))
# screen.blit(car_image, (300, 60))
# screen.blit(RSU_image, (300, 80))
# screen.blit(UAV_image, (300, 100))
# screen.blit(star_image, (300, 120))
# road_image_1 = pygame.transform.rotozoom(car_image, -190, 1)
# screen.blit(road_image_1, (200, 60))
# road_image_2 = pygame.transform.rotozoom(car_image, -45, 1)
# screen.blit(road_image_2, (200, 80))
# road_image_3 = pygame.transform.rotozoom(car_image, 170, 1)
# screen.blit(road_image_3, (200, 100))
# pygame.display.flip()
# pygame.quit()



black = (0, 0, 0)  # 黑色
white = (255, 255, 255)  # 白色
green = (0, 255, 0)  # 绿色
red = (255, 0, 0)  # 红色
blue = (55, 55, 255)  # 蓝色
deepgrey = (125, 125, 125)  # 灰色
grey = (200, 200, 200)  # 灰色
light_grey = (240, 240, 240)  # 灰色
yellow = (255, 185, 15)  # 黄色
orange = (255, 165, 0)  # 橙色
purple = (128, 0, 128)  # 紫色
pink = (255, 192, 203)  # 粉色
forestgreen = (34, 19, 34)  # 森林绿
light_blue=(193,210,240)#浅蓝
light_green=(175, 238, 238)#浅绿
colorlist=[blue,red,green,pink,purple,orange]
colorlistname=['blue','red','green','pink','purple','orange']

# 环境参数
light_speed = 3 * 10 ** 8  # 光速(m/s)
f = 1e+9  # 载波频率（Hz）
power = 1000  # 发射功率(W)
A = 1  # 信道增益系数
B_up = 300  # 10Mbps上行带宽（bps=bit/s）
N0 = 10  # 噪声功率(W)



# 超参数
clip_gs=1#比全局好的奖励限幅
gamma = 0.93  # 折扣因子，越小越喜欢探索
shuaijian = 0.9  # 每一次加奖励时都衰减0.02
hide_num = 32  # 隐藏层神经元个数
# lr = 1e-4
lr = 1e-3




punish_full = 60  # 连接了满负载的RSU惩罚因子
punish_long = 100  # 连接了过远的RSU惩罚因子
punish_satellite = 30#连接卫星时的惩罚
reward_A_same=0#和全局一样时的奖励因子
punish_A_change=0.002#切换一次




pi=np.pi
show = 0
show_A = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_time = 100  # 训练次数
# test_time = 10  # 每100次测试一次
test_time = 100000  # 每100次测试一次
carcsvpath='E:/BaiduNetdiskDownload/data/T_Drive_trajectory/generated_paths.csv'
rsucsvpath='E:/BaiduNetdiskDownload/data/T_Drive_trajectory/beijingrsu.csv'
com_E_work_and_week_all = pd.read_csv('E:/BaiduNetdiskDownload/data/com_E_work_and_week_all.csv')
com_E_work_and_week_all = np.array(com_E_work_and_week_all)
len_data = com_E_work_and_week_all.shape[0]-10000
RSU_num = com_E_work_and_week_all.shape[1]
# 球面距离计算
R = 6371393
Pi = math.pi
# 画图参数
starlocation = [70, 30]
clock = pygame.time.Clock()
FPS = 40  # 帧率设置为60帧每秒
map_size = 800
map_size_2 = map_size * np.sqrt(2)
dis_A = 1#将手画地图等比例放大
show_A_bigger=5 * map_size_2
show_A_bias=100



def get_dis_ball(jingduA, weiduA, jingduB, weiduB):
    a = (torch.sin(Pi / 180 * (weiduA / 2 - weiduB / 2))) ** 2
    b = torch.cos(weiduA * Pi / 180) * torch.cos(weiduB * Pi / 180) * (
        torch.sin((jingduA / 2 - jingduB / 2) * Pi / 180)) ** 2
    dis = 2 * R * torch.asin((a + b) ** 0.5)  # 计算球面距离
    return dis


RSU_jing=0.01#0.0235(2000m)
RSU_wei=0.009#0.018(2000m)
UAV_jing=0.0352#0.0352(3000m)
UAV_wei=0.027#0.027(3000m)
for j in [RSU_range,UAV_range]:
    jingmax = 0.1
    jingmin = 0
    weimax = 0.1
    weimin = 0
    for i in range(100):
        jing=(jingmax+jingmin)/2
        wei=(weimax+weimin)/2
        disjing=get_dis_ball(torch.tensor(0), torch.tensor(40), torch.tensor(jing), torch.tensor(40))
        diswei=get_dis_ball(torch.tensor(0), torch.tensor(40), torch.tensor(0), torch.tensor(40+wei))
        if disjing>j/5:
            jingmax=jing
        else:
            jingmin=jing
        if diswei>j/5:
            weimax=wei
        else:
            weimin=wei
    if j==RSU_range:
        RSU_jing=jing
        RSU_wei=wei
    else:
        UAV_jing = jing
        UAV_wei = wei




# SACLIST = []
# for ppon in range(mappo_num):
#     sacname='AGENT'+str(ppon)
#     SACLIST.append(sacname)
# print(SACLIST)

AGENTLIST = []
for ppon in range(mappo_num):
    pponame = 'AGENT' + str(ppon)
    AGENTLIST.append(pponame)
print(AGENTLIST)


# multicolor = []
# for mrn in range(mappo_num):
#     multicolor.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
# def drawinmain(SACLIST,lis='RR',thing='MASAC',title='time_delay'):
#     save_test_list = []
#     for pponame in SACLIST:
#         if lis=='RR':
#             liss=globals()[pponame].RR
#         elif lis=='EE':
#             liss=globals()[pponame].EE
#         elif lis == 'CC':
#             liss = globals()[pponame].CC
#         elif lis=='JJ':
#             liss=globals()[pponame].JJ
#         elif lis=='DD':
#             liss=globals()[pponame].dropbag
#         save_test_list.append(liss)
#         plt.plot(liss, label=pponame)
#     save_agent(save_test_list, thing+'_'+title + namenum + '.pkl')
#     plt.xlabel('epoch')
#     plt.title(title)
#     plt.legend(loc='best')  # 绘制图例
#     plt.show()

# have_trr=4
# if have_trr==0:
#     generatedata = pd.read_csv('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/generated_paths.csv')
#     beijingdata = pd.read_csv('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/beijingrsu.csv')
#     ###################################################################################
#     # len_car_data, car_data_j, car_data_w = want_car()
#     ###################################################################################
#     rsu_data_j = beijingdata[['lon_j']].values
#     rsu_data_w = beijingdata[['lat_w']].values
#     len_rsu_bj = rsu_data_w.shape[0]
#     ###################################################################################
#     generate_car_j = generatedata[['lon_j']].values
#     generate_car_w = generatedata[['lat_w']].values
#     len_g = len(generatedata)
#     ###################################################################################
#     mini_RSU_j = []
#     mini_RSU_w = []
#     generate_car_j = torch.from_numpy(generate_car_j)
#     generate_car_w = torch.from_numpy(generate_car_w)
#     rsu_data_j = torch.from_numpy(rsu_data_j)
#     rsu_data_w = torch.from_numpy(rsu_data_w)
#     print('数据集RSU个数:',rsu_data_j.shape)
#     for rr in range(len_rsu_bj):  # 去除多余RSU
#         rr_j = rsu_data_j[rr]
#         rr_w = rsu_data_w[rr]
#         well=get_dis_ball(rr_j, rr_w, generate_car_j, generate_car_w) < RSU_range
#         well_index=torch.nonzero(well)
#         if torch.sum(well_index)!=0:#只要距离够，直接保留
#             mini_RSU_j.append(rr_j)
#             mini_RSU_w.append(rr_w)
#     rsu_data_j = torch.tensor(mini_RSU_j)
#     rsu_data_w = torch.tensor(mini_RSU_w)
#     print('筛选后的RSU数量：',rsu_data_j.shape)
#     len_rsu_bj = rsu_data_w.shape[0]
#     ###################################################################################
#     trr = {}  # 存储各个时间点key下的RSU连接车辆列表[1,0,0,1,5,2,0,1]#共有RSU这么长
#     for iwantid in range(1906):  # 车辆id数目共1906
#         start_point = 0  # 开始时候
#         end_point = 0  # 结束时候
#         generate_car_j = generatedata[['lon_j']].values
#         generate_car_w = generatedata[['lat_w']].values
#         id_car = generatedata[['idcar']].values
#         time_car = generatedata[['Time']].values
#         for dd in id_car:
#             dd = dd[0]
#             if iwantid == dd:  # 开始了
#                 end_point += 1
#             else:  # 刚开始还没到想要的日期
#                 if end_point != 0:  # 已经不是第一个开始的了,应该退出了
#                     break
#                 start_point += 1
#         end_point += start_point
#         car_data_j = torch.from_numpy(generate_car_j[start_point:end_point])
#         car_data_w = torch.from_numpy(generate_car_w[start_point:end_point])  # 先取一辆车
#         time_point = time_car[start_point:end_point][:, 0]
#         len_car_data = car_data_w.shape[0]
#         for cc in range(len_car_data):  # 加入该id车辆的所有信息
#             cc_j = torch.tensor(car_data_j[cc])
#             cc_w = torch.tensor(car_data_w[cc])
#             key_trr = time_point[cc]
#             ball_dis=get_dis_ball(cc_j, cc_w, rsu_data_j, rsu_data_w)
#             # put_car = torch.argmin(ball_dis, dim=0)  # 找出最近的RSU
#             put_car = torch.sort(ball_dis, dim=0)  # 找出最近的RSU，最近的RSU排在最前面
#             for pu in put_car[1][:]:
#                 if put_car[0][pu]>RSU_range:#超出服务范围则停止
#                     break
#                 if key_trr not in trr.keys():  # 如果这个时间点是新的，则加入字典
#                     trr[key_trr] = [0] * len_rsu_bj
#                 trr[key_trr][pu] += 1
#     print('时间点:',len(trr))
#     save_agent(trr,'E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/trr.pkl')
#     save_agent(rsu_data_j,'E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/rsu_data_j.pkl')
#     save_agent(rsu_data_w,'E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/rsu_data_w.pkl')
#     save_agent(len_rsu_bj,'E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/len_rsu_bj.pkl')
# elif have_trr==1:
#     trr=load_agent('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/trr.pkl')
#     len_rsu_bj=load_agent('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/len_rsu_bj.pkl')
#     rsu_data_j=load_agent('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/rsu_data_j.pkl')
#     rsu_data_w=load_agent('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/rsu_data_w.pkl')
#     # beijingdata = pd.read_csv('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/beijingrsu.csv')
#     # rsu_data_j = beijingdata[['lon_j']].values
#     # rsu_data_w = beijingdata[['lat_w']].values
#     # rsu_data_j = torch.from_numpy(rsu_data_j)[:, 0]
#     # rsu_data_w = torch.from_numpy(rsu_data_w)[:, 0]
#     rrr={}
#     for num in range(1,10358):
#         # print('num>',num)
#         try:
#             path='E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/06/'+str(num)+'.csv'
#             cardata_time=pd.read_csv(path,usecols=[1]).values
#             cardata_j=pd.read_csv(path,usecols=[2]).values
#             cardata_w=pd.read_csv(path,usecols=[3]).values
#             cardata_j=torch.from_numpy(cardata_j)[:,0]
#             cardata_w=torch.from_numpy(cardata_w)[:,0]
#             len_car_bj=len(cardata_w)
#         except:
#             # print('num>',num)
#             continue
#         for c in range(len_car_bj):
#             ti=cardata_time[c][0]
#             cj=cardata_j[c]
#             wj=cardata_w[c]
#
#             inrange=get_dis_ball(cj, wj, rsu_data_j, rsu_data_w)<=RSU_range
#             inrange_index = torch.nonzero(inrange)
#             if sum(inrange_index)!=0:#要有rsu在范围内才要创建rrr时间点
#                 if ti not in rrr.keys():
#                     rrr[ti] = [0] * len_rsu_bj
#                 for ini in inrange_index[:,0]:
#                     rrr[ti][ini]+=1
#     save_agent(rrr,'E:/BaiduNetdiskDownload/SCI/rrr.pkl')
# elif have_trr==2:
#     smooth=100
#     rrr=load_agent('E:/BaiduNetdiskDownload/SCI/rrr.pkl')
#     len_rsu_bj = load_agent('E:/BaiduNetdiskDownload/SCI/T_Drive_trajectory/len_rsu_bj.pkl')
#     for rsuid in range(len_rsu_bj):
#         data_list = [0] * 1440
#         for key,value in rrr.items():
#             if key[-6]==':':
#                 ho = int(key[-8:-6])
#                 mi = int(key[-5:-3])
#             else:
#                 ho = int(key[-5:-3])
#                 mi = int(key[-2:])
#             data_list[ho * 60 + mi] = value[rsuid]
#         for i in range(1440-smooth):
#             data_list[i]=sum(data_list[i:i+smooth])/smooth
#         save_agent(data_list[:1440-smooth],'E:/BaiduNetdiskDownload/SCI/LSTMpkl/'+str(rsuid)+'.pkl')


caridlist = []
RSUloc = []
UAVloc = []
RR_dis = []
load_RSU = []
pre_load_RSU = []
random_gauss = []
RSUlocal = []
RSUloc_show = []
UAVloc_show = []