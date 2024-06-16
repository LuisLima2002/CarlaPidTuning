
import random
import numpy as np

class Automata:
    def __init__(self,min,max,gw,gh,probabilityStoreSize=1000) -> None:
        self.min = min # minimum value posible for Automata output
        self.max = max # maximum value posible for Automata output
        self.gh = gh # Learning ratio
        self.gw = gw # How large is the affect area
        self.probabilityStoreSize = probabilityStoreSize 
        self.k=0 # number of epochs (inner parameter)
        self.lastAction=0 # (inner parameter)
        self.R=[] # store of J
        
        # self.Z = 0 # use for plot
        
        self.probabilityDistribution=[]  # function f(xi,k)
        self.historyProbabilityDistribution=[]
        initialProb = 1/(self.max-self.min)
        for i in range(self.probabilityStoreSize):
            x=self.min+i*(self.max-self.min)/(self.probabilityStoreSize-1)
            self.probabilityDistribution.append([x,initialProb])
    # select one action, action with more probability are more likely to be choosen 
    def step(self):
        # random between 0-1, action will be the point where
        # the area bellow is equal to z
        z = random.random() 
        
        area=0 # holds currently area ( numeric integral )
        for i in range(self.probabilityStoreSize-1):
            start = self.probabilityDistribution[i][0]
            startValue = self.probabilityDistribution[i][1]
            end = self.probabilityDistribution[i+1][0]
            endValue = self.probabilityDistribution[i+1][1]
            space = np.linspace(start,end,100)
            a = (startValue-endValue)/(start-end)
            b = startValue-start*a
            for j in range(len(space)-1):
                x1 = space[j]
                x2 = space[j+1]
                probability=a*x1+b
                area+=probability*(x2-x1)
                if(abs(area - z)<0.01):
                    self.lastAction=x1
                    return x1
        raise Exception("Not found for z")

    # take a random action between min and max with equal probability
    def explore(self):
        x =  random.random()*(self.max-self.min)+self.min
        self.lastAction=x
        return x


    # uses gaussian function to increase or decrease the probability of one action according to the performance
    def update(self,J):
        self.R.append(J)
        if(len(self.R)<3): return
        if(len(self.R)>500): self.R=self.R[-500:]
        self.Jmed= np.mean(self.R) # medium cost
        self.Jmin= np.min(self.R) # minimum cost
        self.k+=1

        # 0<B<1, if cost is greater than medium it will have a positive
        # impact on the probability, else no impact
        B=np.min([np.max([0.0,(self.Jmed-J)/(self.Jmed-self.Jmin)]),1.0]) 
        
        newprobDistri=[]
        a=0
        for i in range(len(self.probabilityDistribution)):
            x,probability = self.probabilityDistribution[i]

            gh_weighted = (self.gh/(self.max-self.min))

            exp_divider = (2*np.power(self.gw*(self.max-self.min),2))
            
            H=gh_weighted*np.exp(-np.power(x-self.lastAction,2)/exp_divider)
            propNN=probability+B*H
            newprobDistri.append(propNN)
            
            if(i+1<len(self.probabilityDistribution)):
                deltaX =(self.probabilityDistribution[i+1][0]-x)
                deltaY = (self.probabilityDistribution[i+1][1]-propNN)
                a+=propNN*deltaX+deltaY*deltaX/2
        a=1/a # normalise the function
        
        aux = []
        for pair in self.probabilityDistribution:
            aux.append(pair.copy())
        self.historyProbabilityDistribution.append(aux)

        for i in range(len(newprobDistri)):
            self.probabilityDistribution[i][1]=newprobDistri[i]*a
    
    def greatestValue(self):
        maxX = 0
        maxP = 0
        for x,probability in self.probabilityDistribution:
            if(probability>maxP):
                maxP=probability
                maxX=x
        return maxX