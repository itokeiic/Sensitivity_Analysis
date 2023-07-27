
from __future__ import generators
import time
import numpy
    
def FirstOrder(func,bounds,nvar,nout,nsamp=5000):
    """
    This function calculates first order Sobol' indices using two random sample
    sets. First order (ordinary) indices and first order total indices are 
    calculated in nsamp*(2*nvar+2) function evaluations. It is important to note
    that for this calculation to hold input parameters must be uncorrelated to
    each other. cf. Andrea Saltelli, "Making best use of model evaluations to 
    compute sensitivity indices", Computer Physics Communications 145 (2002) 
    280 - 297.
    
    INPUTS:
    func: any python function
    bounds: list of tuples containing lower and upper bounds of input parameters
    nvar: number of input parameters for the func
    nout: number of output responses that you will get from func
    nsamp: number of random samples used in Montecarlo integration
    
    OUTPUTS:
    indices: each row corresponds to a input parameter and each column 
        corresponds to output responses.  The entries are first order 
        indices 
    indicesTot: the total first order indices (includes interactions)
    residues: whatever variance contributions not explained by first order 
        contributions
    Y1: outputs from applying the first sample input set S1 to func
    S1: first Montecarlo sample set
    S2: second Montecarlo sample set
    EE: square of expected values of output responses (if the true expected value 
        is close to 0, EE may become slightly negative due to the approximate nature of 
        sample averaging)
    V: total variance of output responses
    Vj: output variance contribution of the jth input parameter alone (excluding 
        interactions) 
    It also outputs a text file called influence_indices.txt
    """
    lb=[];ub=[]
    for i in range(nvar): #get upper and lower bounds for each parameters
        lb.append(bounds[i][0])
        ub.append(bounds[i][1])
        
    #Set up samples for Montecarlo integration. Here we generate two sets S1,S2.
    #They have nsamp rows and nvar columns
    S_normal1 = numpy.random.rand(nsamp,nvar) # random numbers between 0 and 1
    S_normal2 = numpy.random.rand(nsamp,nvar)
    S1 = numpy.zeros((nsamp,nvar)) # initializing matrices to place
    S2 = numpy.zeros((nsamp,nvar))    
    for i in range(nvar):
        S1[:,i] = S_normal1[:,i] * (ub[i] - lb[i]) + lb[i] # random numbers between lb and ub
        S2[:,i] = S_normal2[:,i] * (ub[i] - lb[i]) + lb[i]

    #Get the ouputs for S1,S2 and compute squared average of outputs and total variance
    Y1 = map(func,S1) #apply func to every row of S1
    Y2 = map(func,S2) #apply func to every row of S2
    Y1 = numpy.array(Y1)
    Y2 = numpy.array(Y2)
    EE = numpy.mean(Y1*Y2,axis=0)
    V = (numpy.var(Y1,axis=0)+numpy.var(Y2,axis=0))/2
    print "Squared average: ", EE
    print "Variance: ", V
    
    #Calculate variance due to parameter j by replacing jth column in S2 with
    #jth column in S1.  Also, calculate the total variace due to parameter j 
    #(contribution of parameter j including interactions of any order) by 
    #replacing jth column in S1 with jth column in S2.
    Vj = []; Vjtot = []
    for j in range(nvar):
        Nj = ColumnReplace(S1,S2,j)
        Uj = numpy.sum(Y1*map(func,Nj),axis=0)/(nsamp-1)
        N_j = ColumnReplace(S2,S1,j)
        U_j = numpy.sum(Y1*map(func,N_j),axis=0)/(nsamp-1)
        vj = Uj - EE # variance due to parameter j
        vjtot = V -(U_j - EE) # variance due to parameter j including interactions
        Vj.append(vj)
        Vjtot.append(vjtot)
    Vj=numpy.array(Vj)
    Vjtot=numpy.array(Vjtot)
    indices=Vj/V
    indicesTot=Vjtot/V
    residues = 1 - indices.sum(0)
    indices=indices.tolist()
    indicesTot=indicesTot.tolist()
#    map(indices.append, indicesTot)# appending indicesTot to indices 

#    indices.append(residues.tolist())
    residues = residues.tolist()
    print 'First order indices:\n',numpy.array(indices)    
    ofile = open("influence_indices.txt","w")
    for i,line in enumerate(indices):
        #ofile.write("[%7d]\t%f\t%f\t%f\t%f\n" %(i,line[0],line[1],line[2],line[3]))
        buf = "[       %3d]" % i
        for j in range(nout):
            buf += "\t%f" % line[j]
        print >> ofile, buf
    for i,line in enumerate(indicesTot):
        #ofile.write("[%7d]\t%f\t%f\t%f\t%f\n" %(i,line[0],line[1],line[2],line[3]))

        buf = "[Tot    %3d]" % i
        for j in range(nout):
            buf += "\t%f" % line[j]
        print >> ofile, buf

    buf = "Higher Order" 
    for j in range(nout):
        buf += "\t%f" % residues[j]
    print >> ofile, buf

    ofile.close()
    print "First order indices written in file: influence_indices.txt"
    return [indices, indicesTot, residues, Y1, S1, S2, EE, V, Vj]
    
def SecondOrder(func, Y1, S1, S2, EE, V, Vj):
    """
    To run this function, it is assumed that you have run FirstOrder.  It 
    computes second order interaction's contribution of the input parameters to
    the output responses. Second order interaction indices and total second 
    order interaction indices are calculated with nsamp*(nvar*(nvar-1)).  As in 
    FirstOrder, the input parameter must be uncorrelated to each other for this
    calculation to be valid.
    
    INPUTS:
    func: any python function (the same one that you used in FirstOrder2)
    for other input arguments refer to FirstOrder
    
    OUTPUTS:
    indices: each row corresponds to a input parameter combination and each 
        column corresponds to output responses.  The entries are second order 
        interaction influence indices. First nvar rows show the second 
    indicesTot: the total second order interaction indices (includes first order
        contributions and interactions of its sub elements with other 
        parameters)
    residues: whatever variance contributions not explained by first or second 
        order contributions
    Adds second order indices to file influence_indices.txt
    """
    
    nvar=len(S1[0])
    nout=len(Y1[0])
    nsamp=len(S1)
    b=xuniqueCombinations(range(nvar),2) # get list of combinations of two parameters
    comb=list(b)
    Vij=[];Vijtot=[]
    for ij in comb: #Need to replace two columns for second order interaction
        Nij = ColumnReplace(S1,S2,ij[0])
        Nij = ColumnReplace(S1,Nij,ij[1])
        Uij = numpy.sum(Y1*map(func,Nij),axis=0)/(nsamp-1)
        N_ij = ColumnReplace(S2,S1,ij[0])
        N_ij = ColumnReplace(S2,N_ij,ij[1])
        U_ij = numpy.sum(Y1*map(func,N_ij),axis=0)/(nsamp-1)
        vij = Uij - EE - Vj[ij[0]] - Vj[ij[1]]
        vijtot = V -(U_ij - EE)
        Vij.append(vij)
        Vijtot.append(vijtot)
    Vij=numpy.array(Vij)
    Vijtot=numpy.array(Vijtot)
    indices=Vij/V
    indicesTot=Vijtot/V
    residues = 1 - indices.sum(0) - numpy.sum(Vj/V,axis=0)
    indices=indices.tolist()
    indicesTot=indicesTot.tolist()
#    map(indices.append, indicesTot)
    indices.append(residues.tolist())
    print '\nSecond order indices:\n',numpy.array(indices)
    
    ofile = open("influence_indices.txt","a")

    for [u,v],line in zip(comb, indices):
        buf="[   %3d,%3d]"%(u,v)
        for j in range(nout):
            buf += "\t%f" % line[j]
        print >> ofile, buf
        
    for [u,v],line in zip(comb, indicesTot):
        buf="[Tot%3d,%3d]"%(u,v)
        for j in range(nout):
            buf += "\t%f" % line[j]
        print >> ofile, buf

    buf = "Higher Order" 
    for j in range(nout):
        buf += "\t%f" % residues[j]
    print >> ofile, buf

    ofile.close()
    print "Second order indices written in file: influence_indices.txt"
    return [indices, indicesTot, residues]

# Utility Functions-------------------------------------------------------------    
def ColumnReplace(M1,M2,j):
    # M1: Column provider
    # M2: Column receiver
    # j: Column indice to be replaced
    # M1 and M2 have identical shape
    Nj = M2.copy()
    Nj[:,j] = M1[:,j]
    return Nj
def xuniqueCombinations(items, n):
    # Calculates Combination with items!/(n!*(items - n)!) elements
    if n==0: 
        yield []
    else:
        for i in xrange(len(items)):
            for cc in xuniqueCombinations(items[i+1:],n-1):
                yield [items[i]]+cc

def func(x):
    #A function example for a test. 3 inputs, 2 outputs
    #return numpy.array([x[0]+2*x[1]+4*x[2],x[0]+x[1]**2+x[0]*x[2]])
    return numpy.array([x[0]+2*x[1]+4*x[2],x[0]**2-x[1]+x[1]*x[2]])

# Test--------------------------------------------------------------------------

if __name__ == "__main__":
    e0=time.time()
    indices1, indicesTot1, residues1, Y1, S1, S2, EE, V, Vj = FirstOrder(func,[(-1,1),(-1,1),(-1,1)],3,2)
    e1=time.time(); lap=e1-e0
    print "Elapsed time for 1st order calculation: ", lap

    e0=time.time()
    indices2, indicesTot2, residues2 = SecondOrder(func, Y1, S1, S2, EE, V, Vj)
    e1=time.time(); lap=e1-e0
    print "Elapsed time for 2nd order calculation: ", lap
"""
The output in "influence_indices.txt" should look something like the following

[         0]	0.043811	0.162731
[         1]	0.186242	0.612578
[         2]	0.745121	0.000992
[Tot      0]	0.068641	0.190833
[Tot      1]	0.211071	0.836323
[Tot      2]	0.769950	0.224737
Higher Order	0.024826	0.223699
[     0,  1]	-0.000003	-0.000046
[     0,  2]	-0.000003	-0.000046
[     1,  2]	-0.000003	0.195597
[Tot  0,  1]	0.254879	0.999008
[Tot  0,  2]	0.813758	0.387422
[Tot  1,  2]	0.956189	0.837269
Higher Order	0.024835	0.028193

The integer number in the first column indicates the parameter index.  In this example,
there are three parameters x_0, x_1, and x_2.  Rows with "Tot" indicates values for 
Total effect indices.  This example has two outputs so there are two columns.
"""