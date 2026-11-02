"""Explorare 26b: strategia „top 10 reguli din istoric cu cele mai scurte pauze”, recalculată după fiecare tură,
pe reguli din explorare26 (n1…n17, n34, n51), în bucăți de 600 de ture (cât jurnalul), de la a 50-a tură a bucății.
Rulare: python cercetare/explorare26b_strategie.py <de_la> <până_la>   (indici de tură, ex. 27617 33617 = ultimele 6.000)"""
import numpy as np
src=open("cercetare/explorare26_pauze_istoric.py").read().split("def gap_stats")[0]
exec(src)
NR=NB*20; Ct=C.T.astype(np.int64)
def play(lo, hi, block=600, start=50):
    res=[]
    for b0 in range(lo,hi,block):
        run=np.zeros(NR,np.int32); mx=run.copy(); hits=run.copy()
        for t in range(b0,min(hi,b0+block)):
            base=X[t]@Ct
            if t-b0>=start:
                key=np.maximum(mx,run).astype(np.int64)*1000000-hits
                ii=np.argpartition(key,10)[:10]; ii=ii[np.lexsort((ii,key[ii]))]
                pr=[int((base[f//20]+f%20-1)%20+1) for f in ii]
                res.append((pr,Y[t]))
            hit=np.arange(NB)*20+(Y[t]-base)%20
            mx=np.maximum(mx,run); run+=1; run[hit]=0; hits[hit]+=1
    return res
import sys
lo,hi=int(sys.argv[1]),int(sys.argv[2]); res=play(lo,hi)
each=np.mean([np.mean([p==y for p in pr]) for pr,y in res]); l1=np.mean([pr[0]==y for pr,y in res])
anyh=np.array([y in set(pr) for pr,y in res]); k=np.mean([len(set(pr)) for pr,_ in res])
r=m=0
for v in anyh: r=0 if v else r+1; m=max(m,r)
print(len(res),"ture; fiecare regula",round(100*each,2),"loc1",round(100*l1,2),"macar una",round(100*anyh.mean(),1),"hazard",round(5*k,1),"numere",round(k,1),"pauza max",m)
