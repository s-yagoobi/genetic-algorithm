


import numpy as np
import networkx as nx
from scipy import linalg 

# Input N: size of graph
# Input W: adjaceny matrix of graph
N=30

G=nx.complete_graph(N)

W=  nx.to_numpy_array(G)







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


W1=np.sum(W, axis=0)
print(W1)
W2=np.reciprocal(W1) #
print(W2)
rho1=0
for row in range(Ntau):
    rho1+=tau[row]*W[ind[row,0],ind[row,1]]/ (W1[ind[row,0]]* W1[ind[row,1]])
    

    
rho1=rho1/(N*np.sum(W2))    
print(rho1)    