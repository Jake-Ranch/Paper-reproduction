import matplotlib.pyplot as plt
import torch.nn

from ELE2MAELE import *

random.seed(513620)
# torch.manual_seed(513620)

L=6400

lr=5e-3
k=7.8*10**(-21)#W/Hz2
tau=0.0001#s

W=10**7#Hz
epision=0.1
gama=0.99
# C=300
# D=2000
p=0.2#W
f=16.8*10**10#Hz本地计算速度
fS=16.8*10**12#HzMEC计算速度
fC=16.8*10**13#Hz云计算速度

# N0=0.001
N0=0.000000001
# N0=0.00000001

beta1=1
beta2=12

psi=6
# psi=0.1

miu=40
v=1

lamda1=1#缓冲区
# lamda1=0.05#缓冲区
lamda2=50#电池惩罚


Emax=30#最大电池容量


totaltime=100

# 数据池
class Pool:
    def __init__(self):
        self.pool = []

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    # 更新动作池
    def update(self):
        # 每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(pool) - old_len < 200:
            self.pool.extend(env.play()[0])
        # 只保留最新的N条数据
        self.pool = self.pool[-1_0000:]

    # 获取一批数据样本
    def sample(self):
        return random.choice(self.pool)


class myenv():
    def __init__(self):
        self.state_dim = 1#len(self.reset())
        self.action_dim = 4
        self.Lm = L  # CPU必要计算周期
        self.pmt = 0.2  # 用户发射功率
        # 定义模型,评估状态下每个动作的价值


    def reset(self):
        # Bmt =  # 当前电池电量
        # emt =  # 充电速度（每秒获得的电能）
        # hmt =  # 概率密度函数，简单起见，大于阈值为1否则为0
        # fmt =   # CPU工作频率
        # Amt = #每秒获得的新任务数据
        self.Qmt = 0  # buffer
        self.Amt = 0
        self.emt = 32.3
        self.hmt = 1
        self.Bmt = Emax  # 初始电量
        self.time = 0
        # state = [Amt, emt, self.hmt, self.Bmt, self.Qmt]  # 任务量Am(t) 能量收集量em(t) 信道增益hm(t) 电池电量Bm(t) 缓冲区中存储的任务量Qm(t)
        # state = 0#[self.Amt, self.emt, self.hmt, self.Bmt, self.Qmt]  # 任务量Am(t) 能量收集量em(t) 信道增益hm(t) 电池电量Bm(t) 缓冲区中存储的任务量Qm(t)
        state = int(min(4,(self.Amt//40000))*(5*5*2*5)+min(4,(self.Bmt//6))*(5*2*5)+min(((self.emt-13.3)//3.8),4)*(2*5)+(self.hmt)*5+min(self.Qmt//40000,4))

        # 任务量Am(t) 能量收集量em(t) 信道增益hm(t) 电池电量Bm(t) 缓冲区中存储的任务量Qm(t)
        return state

    def step(self, state, action, show=False):
        over = 0
        Qmt_ = 0
        Amt = self.Amt
        emt = self.emt
        # Amt, emt=state[:2]
        TmMt = 0
        Amt_ = 0
        if action == 0:  # 本地
            Tmt = (Amt + self.Qmt - Qmt_) * self.Lm / f
            Emt = k * Tmt * (f ** 2)  # 能量消耗
        elif action == 3:  # 不计算，存着
            Qmt_ = Amt + self.Qmt
            Emt = 0
            Tmt = 0
        else:  # 云或MEC
            Amt_ = Amt + self.Qmt
            Tmt = (Amt + self.Qmt - Qmt_) / (
                    wm * math.log2(1 + 0.00001 + self.hmt * self.pmt / (N0 * wm)))  # MEC或Cloud

            Emt = self.pmt * Tmt  # 能量消耗
            if action == 1:  # MEC
                TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
            elif action == 2:  # 云
                TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau

        print(action,
              k * f ** 2 * Tmt,
              (Amt + self.Qmt - Qmt_) * self.Lm / f,
              self.pmt * Tmt,
              self.hmt,
              (Amt + self.Qmt - Qmt_) / (wm * math.log2(1 + 0.00001 + self.hmt * self.pmt / (N0 * wm))),
              (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M),
              (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau
              )

        if self.Bmt + emt - Emt <= 0:
            Qmt_ = Amt + self.Qmt  # 本地计算没电，导致未执行，其大小是Amt+Qmt
            TmMt = 0
            Amt_ = 0
            if action == 0:  # 本地
                Tmt = (Amt + self.Qmt - Qmt_) * self.Lm / f
                Emt = k * Tmt * (f ** 2)  # 能量消耗
            elif action == 3:  # 不计算，存着
                Qmt_ = Amt + self.Qmt
                Emt = 0
                Tmt = 0
            else:  # 云或MEC
                Amt_ = Amt + self.Qmt
                Tmt = (Amt + self.Qmt - Qmt_) / (
                        wm * math.log2(1 + 0.00001 + self.hmt * self.pmt / (N0 * wm)))  # MEC或Cloud

                Emt = self.pmt * Tmt  # 能量消耗
                if action == 1:  # MEC
                    TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
                elif action == 2:  # 云
                    TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau

        Tmt = Tmt + TmMt
        Bmt1 = max(0, min(Emax, self.Bmt + emt - Emt))  # 电池电量
        Cmt = miu * Tmt + v * Emt + lamda1 * self.Qmt / 1000 + lamda2 * (Bmt1 <= 0)  # 包含没电Bmt<0和存储在缓存区的惩罚

        # 上传到云，非本地计算的部分其实就是(Amt+Qmt)*(amM+amL)
        Pmut = abs(Amt - Amt_) * (self.hmt == 1) / 1000
        # 隐私等级，当新收到的数据完全被传到边缘服务器时，就会导致数据被完全解析，因此Amt和Amt_相差越大，隐私越好

        Pmlt = (Amt_ > 0) * (self.hmt == 0)
        # 位置隐私，在信号不好时，用户不上传，那么别人就知道你距离很远，处于一个偏远地区，从而暴露位置

        Pmt = beta1 * Pmut + beta2 * Pmlt

        reward = (psi * Pmt - Cmt) / 1000
        if reward < -100:
            print(self.time, action, 'Bmt', self.Bmt, 'Qmt', self.Qmt / 1000, 'hmt', self.hmt, 'Tmt', Tmt * miu, 'Amt',
                  Amt / 1000, 'Pmt', psi * Pmt, 'Cmt', Cmt, 'Emt', Emt * v, 'reward', reward, 'Pmut', beta1 * Pmut,
                  'Pmlt', beta2 * Pmlt)

        self.Bmt = Bmt1
        self.Qmt = Qmt_
        Amt = random.randint(0, 200 * 1000)
        emt = random.uniform(13.3, 32.3)
        # next_state = [Amt / 1000, emt, self.hmt, self.Bmt, self.Qmt / 1000]
        self.Amt = Amt
        self.emt = emt
        if random.randint(1, 10) > 9:
            self.hmt = 1 - self.hmt  # 0.1概率反转

        # emtEmtlist.append([emt, Emt])
        if self.time >= totaltime:
            over = 1
        self.time += 1
        # over = 0
        # Qmt_ = 0
        # # Amt, emt = state[:2]
        # TmMt = 0
        # Amt_ = 0
        # if action == 0:  # 本地
        #     Tmt = (self.Amt + self.Qmt - Qmt_) * self.Lm / f
        #     Emt = k * f ** 2 * Tmt  # 能量消耗
        # elif action == 3:  # 不计算，存着
        #     Emt = 0
        #     Tmt = 0
        # else:  # 云或MEC
        #     Amt_ = self.Amt + self.Qmt
        #     Tmt = (self.Amt + self.Qmt - Qmt_) / (
        #             wm * math.log2(1 + 0.0001 + self.hmt * self.pmt / (N0 * wm)))  # MEC或Cloud
        #     Emt = self.pmt * Tmt  # 能量消耗
        #     if action == 1:  # MEC
        #         TmMt = (self.Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
        #     elif action == 2:  # 云
        #         TmMt = (self.Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau
        # if self.Bmt + self.emt - Emt <= 0:
        #     Qmt_ = self.Amt + self.Qmt  # 本地计算没电，导致未执行，其大小是Amt+Qmt
        #     TmMt = 0
        #     Amt_ = 0
        #     if action == 0:  # 本地
        #         Tmt = (self.Amt + self.Qmt - Qmt_) * self.Lm / f
        #         Emt = k * f ** 2 * Tmt  # 能量消耗
        #     elif action == 3:  # 不计算，存着
        #         Emt = 0
        #         Tmt = 0
        #     else:  # 云或MEC
        #         Amt_ = self.Amt + self.Qmt
        #         Tmt = (self.Amt + self.Qmt - Qmt_) / (
        #                 wm * math.log2(1 + 0.0001 + self.hmt * self.pmt / (N0 * wm)))  # MEC或Cloud
        #
        #         Emt = self.pmt * Tmt  # 能量消耗
        #         if action == 1:  # MEC
        #             TmMt = (self.Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
        #         elif action == 2:  # 云
        #             TmMt = (self.Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau
        #
        #         # print('MEC分母',TmMt,Tmt, 1 / (wm * math.log2(1 + self.hmt * self.pmt / (N0 * wm) + 0.0001)), self.hmt)
        #
        # Tmt = Tmt + TmMt
        # Bmt1 = max(0, min(Emax, self.Bmt + self.emt - Emt))  # 电池电量
        # Cmt = miu * Tmt + v * Emt + lamda1 * self.Qmt / 1000 + lamda2 * (Bmt1 <= 0)  # 包含没电Bmt<0和存储在缓存区的惩罚
        #
        # # 上传到云，非本地计算的部分其实就是(Amt+Qmt)*(amM+amL)
        # Pmut = abs(self.Amt - Amt_) * (self.hmt == 1) / 1000
        # # 隐私等级，当新收到的数据完全被传到边缘服务器时，就会导致数据被完全解析，因此Amt和Amt_相差越大，隐私越好
        #
        # Pmlt = (Amt_ > 0) * (self.hmt == 0)
        # # 位置隐私，在信号不好时，用户不上传，那么别人就知道你距离很远，处于一个偏远地区，从而暴露位置
        #
        # Pmt = beta1 * Pmut + beta2 * Pmlt
        #
        # reward = psi * Pmt - Cmt
        # if reward < -100:
        #     print(self.time, action, 'Bmt', self.Bmt, 'Qmt', self.Qmt / 1000, 'hmt', self.hmt, 'Tmt', Tmt * miu, 'Amt',
        #           self.Amt / 1000, 'Pmt', psi * Pmt, 'Cmt', Cmt, 'Emt', Emt * v, 'reward', reward, 'Pmut', beta1 * Pmut,
        #           'Pmlt', beta2 * Pmlt)
        #
        # self.Bmt = Bmt1
        # self.Qmt = Qmt_
        # Amt = random.randint(0, 200 * 1000)
        # emt = random.uniform(13.3, 32.3)
        # next_state = [Amt / 1000, emt, self.hmt, self.Bmt, self.Qmt / 1000]
        # if random.randint(1, 10) > 9:
        #     self.hmt = 1 - self.hmt  # 0.1概率反转
        #
        # emtEmtlist.append([emt, Emt])
        # if self.time >= totaltime:
        #     over = 1
        # self.time += 1
        # # print(state,next_state)
        # self.Amt=Amt
        # self.emt=emt
        next_state=int(min(4,(self.Amt//40000))*(5*5*2*5)+min(4,(self.Bmt//6))*(5*2*5)+min(((self.emt-13.3)//3.8),4)*(2*5)+(self.hmt)*5+min(self.Qmt//40000,4))
        return next_state, reward, over

    # 玩一局游戏并记录数据
    # def play(self,show=False):
    #     global emtEmtlist
    #     data = []#state, action, reward, next_state, over
    #     reward_sum = 0
    #
    #     if show:
    #         emtEmtlist=[]
    #     state = env.reset()
    #     over = False
    #     while not over:
    #         action = self.model(torch.FloatTensor(state).reshape(1, state_dim)).argmax().item()
    #         if random.random() < 0.1:
    #             action = random.randint(0,self.action_dim-1)
    #         with torch.no_grad():
    #             actionbar[action]+=1
    #         next_state, reward, over = env.step(state,action,show=show)
    #
    #         data.append((state, action, reward, next_state, over))
    #         reward_sum += reward
    #
    #         state = next_state
    #     save_agent(emtEmtlist,str(L)+'emtEmtlist.pkl')
    #     return data, reward_sum

    # 玩一局游戏并记录数据
    def play(self, show=False):
        global emtEmtlist
        data = []
        reward_sum = 0
        # if show:
        #     emtEmtlist = []
        state = env.reset()
        over = False
        while not over:
            action = Q[state].argmax()
            if random.random() < 0.1:
                action = random.randint(0, self.action_dim - 1)
            with torch.no_grad():
                actionbar[action] += 1

            next_state, reward, over = env.step(state, action, show=show)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state
        # save_agent(emtEmtlist, str(L) + 'emtEmtlist.pkl')

        return data, reward_sum


# pool = Pool()
# pool.update()
pool = Pool()
env = myenv()
losslist = []

test_resultlist = []


# 训练

# def train():
#     env.model.train()
#     # optimizer = torch.optim.Adam(env.model.parameters(), lr=1e-3)
#     optimizer = torch.optim.SGD(env.model.parameters(), lr=1e-4)
#     loss_fn = torch.nn.MSELoss()
#
#     # 共更新N轮数据
#     for epoch in range(101):
#         pool.update()
#         losssum = 0
#         # 每次更新数据后,训练N次
#         for i in range(200):
#             # 采样N条数据
#             state, action, reward, next_state, over = pool.sample()
#
#             # 计算value
#             value = env.model(state).gather(dim=1, index=action)
#
#             # print(state)
#             # 计算target
#             with torch.no_grad():
#                 target = env.model(next_state)
#             target = target.max(dim=1)[0].reshape(-1, 1)
#             target = target * 0.99 * (1 - over) + reward
#
#             loss = loss_fn(value, target)
#
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             losssum += loss
#             # print(value, target)
#
#             # if epoch % 10 == 0:
#             test_result = sum(reward)[0]
#             # test_result = sum([env.play()[-1] for _ in range(20)]) / 20
#             # print(epoch, len(pool), test_result)
#             test_resultlist.append(test_result)
#
#         print(losssum / 200)
#         losslist.append(losssum / 200)
#         if epoch % 10 == 0:
#             print(epoch, len(pool), sum([env.play(show=True)[-1] for _ in range(20)]) / 20)


# 训练
def train1():
    # 共更新N轮数据
    # for epoch in range(20000):
    # for epoch in range(1000):
    for epoch in range(6400):
        pool.update()
        losssum = 0
        # 每次更新数据后,训练N次
        for i in range(200):

            # 随机抽一条数据
            state, action, reward, next_state, over = pool.sample()

            # Q矩阵当前估计的state下action的价值
            value = Q[state, action]

            # 实际玩了之后得到的reward+下一个状态的价值*0.9
            target = reward + Q[next_state].max() * 0.9

            # value和target应该是相等的,说明Q矩阵的评估准确
            # 如果有误差,则应该以target为准更新Q表,修正它的偏差
            # 这就是TD误差,指评估值之间的偏差,以实际成分高的评估为准进行修正
            update = (target - value) * 0.01

            # 更新Q表
            Q[state, action] += update

            losssum += update
            print(update)

            test_result = reward
            test_resultlist.append(test_result)

        print(losssum / 200)
        losslist.append(losssum / 200)

        if epoch % 100 == 0:
            print(epoch, len(pool), env.play()[-1])


state_dim = env.state_dim
action_dim = env.action_dim  # 本地、edgeserver cloud buffer save

# 初始化Q表,定义了每个状态下每个动作的价值
Q = np.zeros((5 * 5 * 5 * 5 * 2, action_dim))
# for At in range(0,):


M = 4  # 4个用户

wm = W / M  # 平均带宽，W为总带宽，m为用户数量
with torch.no_grad():
    actionbar = [0] * action_dim
train1()

with torch.no_grad():

    save_agent(test_resultlist, str(L) + 'Qtest_resultlist.pkl')
    save_agent(losslist, str(L) + 'Qlosslist.pkl')
    save_agent(actionbar, str(L) + 'Qactionbar.pkl')
    plt.plot(actionbar)
    plt.show()
    plt.plot(test_resultlist)
    plt.show()
