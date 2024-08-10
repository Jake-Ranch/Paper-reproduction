import matplotlib.pyplot as plt
import torch.nn

from ELE2MAELE import *
L=0
random.seed(513620)
# torch.manual_seed(513620)



lr=3e-5
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
N0=0.0000000001
# N0=0.000000001

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









#数据池
class Pool:

    def __init__(self):
        self.pool = []

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    #更新动作池
    def update(self):
        #每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(pool) - old_len < 300:
            self.pool.extend(env.play()[0])
            # self.pool.extend(env.play())
        #只保留最新的N条数据
        self.pool = self.pool[-2_0000:]

    #获取一批数据样本
    def sample(self):
        data = random.sample(self.pool, 64)
        state = torch.FloatTensor([i[0] for i in data]).reshape(-1, state_dim)
        action = torch.LongTensor([i[1] for i in data]).reshape(-1, 1)
        reward = torch.FloatTensor([i[2] for i in data]).reshape(-1, 1)
        next_state = torch.FloatTensor([i[3] for i in data]).reshape(-1, state_dim)
        over = torch.LongTensor([i[4] for i in data]).reshape(-1, 1)
        return state, action, reward, next_state, over


class myenv():
    def __init__(self):
        self.state_dim=len(self.reset())
        self.action_dim=4
        self.Lm = L # CPU必要计算周期
        self.pmt = 0.2  # 用户发射功率
        # 定义模型,评估状态下每个动作的价值
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            # torch.nn.ReLU(),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, self.action_dim),
        )

    def reset(self):
        # Bmt =  # 当前电池电量
        # emt =  # 充电速度（每秒获得的电能）
        # hmt =  # 概率密度函数，简单起见，大于阈值为1否则为0
        # fmt =   # CPU工作频率
        # Amt = #每秒获得的新任务数据
        self.Qmt=0#buffer
        self.Amt=0
        self.emt=32.3
        self.hmt=1
        self.Bmt=Emax#初始电量
        self.time=0
        state=[self.Amt/1000,self.emt,self.hmt,self.Bmt,self.Qmt/1000]#任务量Am(t) 能量收集量em(t) 信道增益hm(t) 电池电量Bm(t) 缓冲区中存储的任务量Qm(t)
        return state

    def step(self,state,action,show=False):
        over=0
        Qmt_ =0
        Amt=self.Amt
        emt=self.emt
        # Amt, emt=state[:2]
        TmMt = 0
        Amt_=0
        if action == 0:  # 本地
            Tmt = (Amt + self.Qmt - Qmt_) * self.Lm / f
            Emt = k  * Tmt *(f ** 2)  # 能量消耗
        elif action == 3:  # 不计算，存着
            Qmt_=Amt+self.Qmt
            Emt = 0
            Tmt = 0
        else:  # 云或MEC
            Amt_ = Amt + self.Qmt
            Tmt = (Amt + self.Qmt - Qmt_) / (
                    wm * math.log2(1+ 0.00001 + self.hmt * self.pmt / (N0 * wm)) )  # MEC或Cloud

            Emt = self.pmt * Tmt  # 能量消耗
            if action == 1:  # MEC
                TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
            elif action == 2:  # 云
                TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau

        # print(action,
        #       k * f ** 2 * Tmt,
        #       (Amt + self.Qmt - Qmt_) * self.Lm / f,
        #       self.pmt * Tmt,
        #       self.hmt,
        #       (Amt + self.Qmt - Qmt_) / (wm * math.log2(1+ 0.00001 + self.hmt * self.pmt / (N0 * wm))),
        #       (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M),
        #       (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau
        #       )

        if self.Bmt+emt-Emt<=0:
            Qmt_ =  Amt+self.Qmt# 本地计算没电，导致未执行，其大小是Amt+Qmt
            TmMt = 0
            Amt_=0
            if action == 0:  # 本地
                Tmt = (Amt + self.Qmt - Qmt_) * self.Lm / f
                Emt = k  * Tmt *(f ** 2)  # 能量消耗
            elif action==3:#不计算，存着
                Qmt_ = Amt + self.Qmt
                Emt=0
                Tmt=0
            else:  # 云或MEC
                Amt_ = Amt + self.Qmt
                Tmt = (Amt + self.Qmt - Qmt_) / (
                            wm * math.log2(1+ 0.00001 + self.hmt * self.pmt / (N0 * wm)) )  # MEC或Cloud

                Emt = self.pmt * Tmt  # 能量消耗
                if action == 1:  # MEC
                    TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fS / M)  # M个用户发给云
                elif action==2:  # 云
                    TmMt = (Amt + self.Qmt - Qmt_) * self.Lm / (fC / M) + tau


        Tmt = Tmt + TmMt
        Bmt1=max(0,min(Emax,self.Bmt+emt-Emt))#电池电量
        Cmt=miu*Tmt+v*Emt+lamda1*self.Qmt/1000+lamda2*(Bmt1<=0)#包含没电Bmt<0和存储在缓存区的惩罚

        #上传到云，非本地计算的部分其实就是(Amt+Qmt)*(amM+amL)
        Pmut=abs(Amt-Amt_)*(self.hmt==1)/1000
        #隐私等级，当新收到的数据完全被传到边缘服务器时，就会导致数据被完全解析，因此Amt和Amt_相差越大，隐私越好

        Pmlt=(Amt_>0)*(self.hmt==0)
        #位置隐私，在信号不好时，用户不上传，那么别人就知道你距离很远，处于一个偏远地区，从而暴露位置

        Pmt=beta1*Pmut+beta2*Pmlt



        reward=(psi*Pmt-Cmt)/1000
        # if reward<-100:print(self.time,action,'Bmt',self.Bmt,'Qmt',self.Qmt/1000,'hmt',self.hmt,'Tmt',Tmt*miu,'Amt',Amt/1000,'Pmt',psi*Pmt,'Cmt',Cmt,'Emt',Emt*v,'reward',reward,'Pmut',beta1*Pmut,'Pmlt',beta2*Pmlt)

        self.Bmt=Bmt1
        self.Qmt=Qmt_
        Amt=random.randint(0,200*1000)
        emt=random.uniform(13.3,32.3)
        next_state=[Amt/1000,emt,self.hmt,self.Bmt,self.Qmt/1000]
        self.Amt = Amt
        self.emt = emt
        if random.randint(1,10)>9:
            self.hmt=1-self.hmt#0.1概率反转

        emtEmtlist.append([emt,Emt])
        if self.time>=totaltime:
            over=1
        self.time+=1
        Privacylist.append(Pmt)
        Utilitylist.append(reward)
        Latencylist.append(Tmt)
        Energylist.append(Emt)

        return next_state, reward, over


    # 玩一局游戏并记录数据
    def play(self,show=False):
        global emtEmtlist,Privacylist,Utilitylist,Latencylist,Energylist
        data = []#state, action, reward, next_state, over
        reward_sum = 0

        if show:
            emtEmtlist=[]
            Privacylist=[]
            Utilitylist=[]
            Latencylist=[]
            Energylist=[]
        state = env.reset()
        over = False
        while not over:
            action = self.model(torch.FloatTensor(state).reshape(1, state_dim)).argmax().item()
            if random.random() < epision:
                action = random.randint(0,self.action_dim-1)
            with torch.no_grad():
                actionbar[action]+=1
            next_state, reward, over = env.step(state,action,show=show)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state
        save_agent(emtEmtlist,str(L)+'emtEmtlist.pkl')
        return data, reward_sum



#训练
def train():
    env.model.train()
    # optimizer = torch.optim.Adam(env.model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(env.model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    #共更新N轮数据
    for epoch in range(101):
        pool.update()
        losssum=0
        #每次更新数据后,训练N次
        for i in range(200):

            #采样N条数据
            state, action, reward, next_state, over = pool.sample()

            #计算value
            value = env.model(state).gather(dim=1, index=action)

            with torch.no_grad():
                target = env.model(next_state)
            target = target.max(dim=1)[0].reshape(-1, 1)
            target = target * gama * (1 - over) + reward

            loss = loss_fn(value, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losssum+=loss


            test_result = sum(reward)[0]
            test_resultlist.append(test_result)

        print(losssum / 200)
        losslist.append(losssum / 200)
        if epoch % 10 == 0:
            print(epoch, len(pool), sum([env.play(show=True)[-1] for _ in range(20)]) / 20)

# for L in [10000,15000,20000,25000,30000]:
for L in range(10000,30000,1500):
# for M in [1,2,3,4]:
# if 1:
#     L=6400
    # L=10000
    # L=15000
    # L=20000
    # L=25000
    # L=30000
    M=4#4个用户
    if L<20000:
        psi=6
        # N0=0.0000000001
    else:
        # N0=0.0000001
        # psi=0.6
        psi=0.1

    wm = W / M  # 平均带宽，W为总带宽，m为用户数量

    emtEmtlist = []
    Privacylist = []
    Utilitylist = []
    Latencylist = []
    Energylist = []

    pool = Pool()
    env=myenv()
    losslist=[]

    test_resultlist=[]

    state_dim=env.state_dim

    action_dim=env.action_dim#本地、edgeserver cloud buffer save
    with torch.no_grad():
        actionbar=[0]*action_dim
    train()

    with torch.no_grad():
        save_agent(test_resultlist,str(L)+'test_resultlist.pkl')
        save_agent(Energylist,str(L)+'Energylist.pkl')
        save_agent(Utilitylist,str(L)+'Utilitylist.pkl')
        save_agent(Latencylist,str(L)+'Latencylist.pkl')
        save_agent(Privacylist,str(L)+'Privacylist.pkl')
        save_agent(losslist,str(L)+'losslist.pkl')
        save_agent(actionbar,str(L)+'actionbar.pkl')
        if 0:
            plt.plot(test_resultlist)
            plt.title('test_resultlist')
            plt.show()
            plt.plot(losslist)
            plt.title('losslist')
            plt.show()
            plt.plot(actionbar)
            plt.title('actionbar')
            plt.show()
            plt.plot(Utilitylist)
            plt.title('Utilitylist')
            plt.show()
            plt.plot(Latencylist)
            plt.title('Latencylist')
            plt.show()
            plt.plot(Energylist)
            plt.title('Energylist')
            plt.show()
            plt.plot(Privacylist)
            plt.title('Privacylist')
            plt.show()




