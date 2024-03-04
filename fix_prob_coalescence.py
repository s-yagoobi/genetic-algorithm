


import numpy as np
import networkx as nx
from scipy import linalg 
import matplotlib.pyplot as plt

# Input N: size of graph
# Input W: adjaceny matrix of graph

def comet(N,m):
    K=np.zeros([N,N])
    for i in range(m):
        K[i,0:m]=1
        K[i,i]=0
    K[0,m:N]=1
    K[m:N,0]=1
    return K


def star_line(N,m):
    SL=np.zeros([N,N])
    SL[0:m,m-1]=1
    SL[m-1,0:m]=1
    SL[m-1,m-1]=0
    for i in range(m,N-1):
        SL[i,i-1]=1
        SL[i,i+1]=1
        SL[i-1,i]=1
        SL[i+1,i]=1
    SL[N-1,N-2]=1
    
    return SL    
        
#G=nx.complete_graph(N)

#W=  nx.to_numpy_array(G)
'''W=np.zeros([N,N])

for i in range(N):
    W[0,:]=1
    W[:,0]=1
W[0,0]=1  '''










        
        

def Tau(Ntau):
    # Define variables
    Ntau =int( N*(N-1)/2 )     # number of unique taus (remeeting times) to solve
    wdeg = np.sum(W,axis=1)        # weighted degree

    wdeg1=np.zeros([N,N])
    for i in range(N):
        wdeg1[i,:]=wdeg
    
    wdeg1=np.transpose(wdeg1)

    #one step probability
    P=np.divide(W,wdeg1)

    #wdeginv = 1./wdeg      # reciprocal of weigthed degree
    #Wtilde = sum(wdeginv)  # sum of inverse weigthed degrees


    Temp = np.sum(P,axis=0)       # node temperature
    ##Solve for taus (coalescence times)
    #Determine indices of non-zero taus
    ind = np.zeros([Ntau,2],int)
    k = 0
    for i in range(N):
        for j in range(i+1,N):
        
            ind[k,:]= (i,j)
            k = k + 1


    # Determine A coefficient matrix
    A = np.zeros([Ntau,Ntau])

    #Determine b constant matrix
    b=np.zeros([Ntau])

    for row in range(Ntau):
    
        b[row]=2/(Temp[ind[row,0]]+Temp[ind[row,1]])
        for col in range(row + 1 , Ntau):
        
            if ind[row,0] == ind[col,0]:
                A[row,col] = P [ind[col,1],ind[row,1]]/(Temp[ind[row,0]]+Temp[ind[row,1]])
                A[col,row] = P [ind[row,1],ind[col,1]]/(Temp[ind[col,0]]+Temp[ind[col,1]])
            
            elif (ind[row,1] == ind[col,0]):
                A[row,col] = P [ind[col,1],ind[row,0]]/(Temp[ind[row,0]]+Temp[ind[row,1]])
                A[col,row] = P [ind[row,0],ind[col,1]]/(Temp[ind[col,0]]+Temp[ind[col,1]])
            elif (ind[row,1] == ind[col,1]):
                A[row,col] = P [ind[col,0],ind[row,0]]/(Temp[ind[row,0]]+Temp[ind[row,1]])
                A[col,row] = P [ind[row,0],ind[col,0]]/(Temp[ind[col,0]]+Temp[ind[col,1]])
            

    #calculate coalescence time
    tau=np.matmul(b, linalg.inv(np.identity(Ntau)-A))
    return tau

def fixation_second(W,N):
    Ntau =int( N*(N-1)/2 )
    ind = np.zeros([Ntau,2],int)
    k = 0
    for i in range(N):
        for j in range(i+1,N):
        
            ind[k,:]= (i,j)
            k = k + 1
    W1=np.sum(W, axis=0) #sum of the  weight of outgoing edges

    W2=np.reciprocal(W1) #the inverse of entries of W1
    tau=Tau(Ntau)
    rho1=0
    for row in range(Ntau):
        rho1+=tau[row]*W[ind[row,0],ind[row,1]]/ (W1[ind[row,0]]* W1[ind[row,1]])
    

    #the linear term term of fixation probability    
    rho1=rho1/(N*np.sum(W2))    
    return rho1

f=open('cometN_200.txt','w')

Second_order=[]
N=200 
for m in range(1,N):
    W= comet(N,m)   
    Second_order.append(fixation_second(W,N))
    f.write(str(m)+'\t'+str(fixation_second(W, N))+'\n')
f.close()    
#plt.plot(Second_order)    
#plt.show()