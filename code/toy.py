import sys, itertools, numpy as np

from gencg import learn, objective
F="abcde"
D=[([1,1,1,1,0],1),([1,1,1,0,1],1),([1,1,0,0,1],1),([1,0,0,0,1],1),
   ([1,0,0,1,1],0),([1,1,0,0,0],0),([0,1,0,1,0],0),([0,1,0,1,1],0)]
B=np.array([x for x,_ in D],dtype=np.uint8); y=np.array([l for _,l in D])
name=lambda X:' ∧ '.join(('' if v else '¬')+F[f] for f,v in X)
for fr in (False,True):
    res=learn(B,y,k=2,lam=10,time_limit=30,freeness=fr)
    print("freeness",fr,"rule:",' ∨ '.join('('+name(X)+')' for X in res['rule']),"f=",objective(B,y,res['rule'],10),
          "it",res['iterations'],"pool",res['pool'],"certified",res['certified'],"oracle calls",res['n_oracle'])
    for t in res['trace']:
        print("   it",t['it'],"R=",[name(X) for X in t['R']],"improving:",[(('add' if j is None else f'replace {name(t["R"][j])}'),name(X),g,T) for j,X,g,T in t['found']])
