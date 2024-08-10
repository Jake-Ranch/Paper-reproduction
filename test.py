import matplotlib.pyplot as plt
from ELE2MAELE import *
def mean(list1):
    return sum(list1)/len(list1)

test=[]
for M in [1,2,3]:
    test_resultlist=mean(load_agent(str(M)+'test_resultlist.pkl'))
    Qtest_resultlist=sum((load_agent(str(M)+'Qtest_resultlist.pkl'))[:32]+(load_agent(str(M)+'Qtest_resultlist.pkl'))[-32:])
    test.append(Qtest_resultlist)
    test.append(test_resultlist)
colors = ['blue','orange']*3
timeaxis=['Q1','D1','Q2','D2','Q3','D3',]
plt.bar(timeaxis,test,color=colors,width=0.7)
plt.ylabel('Reward')
plt.xlabel('Number of Users devices')
plt.show()



Energy=[]
Utility=[]
Latency=[]
Privacy=[]
LEnergy=[]
LUtility=[]
LLatency=[]
LPrivacy=[]
OEnergy=[]
OUtility=[]
OLatency=[]
OPrivacy=[]
QEnergy=[]
QUtility=[]
QLatency=[]
QPrivacy=[]
for L in range(10000,30000,1500):
    Energylist=load_agent(str(L)+'Energylist.pkl')
    Utilitylist=load_agent(str(L)+'Utilitylist.pkl')
    Latencylist=load_agent(str(L)+'Latencylist.pkl')
    Privacylist=load_agent(str(L)+'Privacylist.pkl')
    Energy.append(mean(Energylist))
    Utility.append(mean(Utilitylist))
    Latency.append(mean(Latencylist))
    Privacy.append(mean(Privacylist))

    OEnergylist=load_agent(str(L)+'OEnergylist.pkl')
    OUtilitylist=load_agent(str(L)+'OUtilitylist.pkl')
    OLatencylist=load_agent(str(L)+'OLatencylist.pkl')
    OPrivacylist=load_agent(str(L)+'OPrivacylist.pkl')
    OEnergy.append(mean(OEnergylist))
    OUtility.append(mean(OUtilitylist))
    OLatency.append(mean(OLatencylist))
    OPrivacy.append(mean(OPrivacylist))

    QEnergylist=load_agent(str(L)+'QEnergylist.pkl')
    QUtilitylist=load_agent(str(L)+'QUtilitylist.pkl')
    QLatencylist=load_agent(str(L)+'QLatencylist.pkl')
    QPrivacylist=load_agent(str(L)+'QPrivacylist.pkl')
    QEnergy.append(mean(QEnergylist))
    QUtility.append(mean(QUtilitylist))
    QLatency.append(mean(QLatencylist))
    QPrivacy.append(mean(QPrivacylist))

    LEnergylist=load_agent(str(L)+'LEnergylist.pkl')
    LUtilitylist=load_agent(str(L)+'LUtilitylist.pkl')
    LLatencylist=load_agent(str(L)+'LLatencylist.pkl')
    LPrivacylist=load_agent(str(L)+'LPrivacylist.pkl')
    LEnergy.append(mean(LEnergylist))
    LUtility.append(mean(LUtilitylist))
    LLatency.append(mean(LLatencylist))
    LPrivacy.append(mean(LPrivacylist))
timeaxis=torch.linspace(1,3,len(Energy))


plt.plot(timeaxis,Utility,label='DQN')
plt.plot(timeaxis,OUtility,label='Offloading Execution')
plt.plot(timeaxis,LUtility,label='Local Execution')
plt.plot(timeaxis,QUtility,label='Q-Learning')
plt.title('Utility')
plt.xlabel('L(x10000 Cycles/bit)')
plt.legend()
plt.show()

plt.plot(timeaxis,Latency,label='DQN')
plt.plot(timeaxis,OLatency,label='Offloading Execution')
plt.plot(timeaxis,LLatency,label='Local Execution')
plt.plot(timeaxis,QLatency,label='Q-Learning')
plt.title('Latency(ms)')
plt.xlabel('L(x10000 Cycles/bit)')
plt.legend()
plt.show()


plt.plot(timeaxis,Energy,label='DQN')
plt.plot(timeaxis,OEnergy,label='Offloading Execution')
plt.plot(timeaxis,LEnergy,label='Local Execution')
plt.plot(timeaxis,QEnergy,label='Q-Learning')
plt.title('Energy Consumption(J)')
plt.xlabel('L(x10000 Cycles/bit)')
plt.legend()
plt.show()

plt.plot(timeaxis,Privacy,label='DQN')
plt.plot(timeaxis,OPrivacy,label='Offloading Execution')
plt.plot(timeaxis,LPrivacy,label='Local Execution')
plt.plot(timeaxis,QPrivacy,label='Q-Learning')
plt.title('Privacy level')
plt.xlabel('L(x10000 Cycles/bit)')
plt.legend()
plt.show()



# timeaxisbar=['Local','MEC','Cloud','Save in Buffer']
timeaxisbar=['B\n','L\n','M\n','C\n',]
timeaxis=[]
Llist=[10000,15000,20000,25000,30000]
global actionbar
for L in Llist:
    actionbarlist=load_agent(str(L)+'actionbar.pkl')
    # actionbarlist[0], actionbarlist[3] = actionbarlist[3], actionbarlist[0]
    actionbarlist.insert(0,actionbarlist[-1])
    actionbarlist=actionbarlist[:4]
    if L==Llist[0]:
        actionbar=torch.tensor(actionbarlist)/5.2823
    else:
        actionbar = torch.cat([actionbar,torch.tensor(actionbarlist) / 5.2823],dim=0)
    for i in timeaxisbar:
        timeaxis.append(i+str(L))#*len(Llist)
colors = ['orange','blue', 'green', 'red', ]*len(Llist)
plt.grid()
plt.bar(timeaxis,actionbar,width=1,color=colors)
# print(sum(actionbar),len(timeaxis),len(colors),actionbar,timeaxis)
plt.ylabel('action(times)')
plt.show()
L=6400



test_resultlist=load_agent(str(L)+'test_resultlist.pkl')
showlist=[]
for i in range(len(test_resultlist)//200):
    # showlist.append(test_resultlist[200*i])
    showlist.append(np.mean(test_resultlist[200*i:200*(i+1)]))
timeaxis=torch.linspace(0,10000,len(showlist))
plt.plot(timeaxis,showlist,label='Deep RL Offloading',color='orange',linestyle='--',marker='o')
test_resultlist=load_agent(str(L)+'Qtest_resultlist.pkl')
showlist=[]
showlist1=[]
# plt.plot(test_resultlist)
for i in range(len(test_resultlist)//64):
    showlist.append(sum(test_resultlist[64*i:64*(i+1)]))
# for i in range(len(showlist)-100):
#     showlist1.append(np.mean(showlist[i:(i+100)]))
for i in range(len(showlist)//200):
    showlist1.append(np.mean(showlist[i*200:200*(i+1)]))
timeaxis=torch.linspace(0,10000,len(showlist1))
plt.plot(timeaxis,showlist1,label='Q-learning Offloading',color='blue',linestyle='--',marker='*',alpha=0.5)
plt.ylabel('Utility(Reward)')
plt.xlabel('Time Slot')
plt.grid()
plt.legend()
plt.show()


emtEmtlist=torch.tensor(load_agent(str(L)+'emtEmtlist.pkl'))
timeaxis=torch.linspace(1,emtEmtlist.shape[0],emtEmtlist.shape[0])
plt.bar(timeaxis,emtEmtlist[:,0],color='green',label='Harvested Energy',width=1)
plt.bar(timeaxis,-emtEmtlist[:,1],color='red',label='Energy Cost',width=1)
plt.ylabel('Energy(J)')
plt.xlabel('Time Slot')
plt.legend()
plt.show()

