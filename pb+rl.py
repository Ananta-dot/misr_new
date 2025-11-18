#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatternBoost (optimized) — MISR worst-case search using exact LP/ILP (Gurobi).

What changed vs your previous PB:
- Annealed bias: per-round α,β,γ,λ schedule -> pure-ish ratio late in each n.
- Novelty elites: keep diverse top patterns by normalized H/V Hamming distance.
- Dual-guided neighbors: edits focused on hottest spans & corners.
- Stagnation handling: occasional heavy shake when blended score stalls.
- Dual-aware lift: insert new labels at hot indices to preserve bad structure.
- Seed remix: recombine elites + strong jitter, then novelty filter.
- Rich logging: q50/q75/q90/q99 and plateau warnings.

All still “PatternBoost only” (no separate RL trainer).
"""

from __future__ import annotations

import argparse, math, os, random, sys, time, tempfile, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---- torch
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch is required. `pip install torch`", file=sys.stderr); raise

# ---- gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    print("ERROR: gurobipy + license required.", file=sys.stderr); raise

# =========================
# Device (MPS friendly)
# =========================
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# =========================
# Types & utils
# =========================
Seq = List[int]; Instance = Tuple[Seq,Seq]
SPECIAL={"BOS":0,"SEP":1,"EOS":2}; BASE_VOCAB=3; MAX_N=256

def seq_spans(seq: Seq) -> List[Tuple[int,int]]:
    first,spans={},{}
    for idx,lab in enumerate(seq):
        if lab not in first: first[lab]=idx
        else: spans[lab]=(first[lab],idx)
    n=max(seq) if seq else 0
    return [spans[i] for i in range(1,n+1)]

def canonicalize(H:Seq,V:Seq)->Instance:
    order=[]; seen=set()
    for x in H:
        if x not in seen: order.append(x); seen.add(x)
    rel={old:new for new,old in enumerate(order,1)}
    return [rel[x] for x in H],[rel[x] for x in V]

def instance_key(H:Seq,V:Seq)->str:
    s=",".join(map(str,H))+"|"+",".join(map(str,V))
    return hashlib.blake2b(s.encode(),digest_size=16).hexdigest()

def random_valid_seq(n:int,rng:random.Random)->Seq:
    seq=[i for i in range(1,n+1) for _ in range(2)]; rng.shuffle(seq); return seq

# motifs
def motif_rainbow(n:int)->Seq: return list(range(1,n+1))+list(range(n,0,-1))
def motif_doubled(n:int)->Seq: return [x for i in range(1,n+1) for x in (i,i)]
def motif_interleave(n:int)->Seq:
    out=[]; 
    for i in range(1,n+1,2):
        j=i+1 if i+1<=n else i; out+=[i,j,i,j]
    # fix counts
    cnt={i:0 for i in range(1,n+1)}; fixed=[]
    for x in out:
        if cnt[x]<2: fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i]<2: fixed.append(i); cnt[i]+=1
    return fixed[:2*n]
def motif_zipper(n:int)->Seq:
    out=[]
    for i in range(1,(n//2)+1):
        j=n-i+1; out+=[i,j,i,j]
    if n%2==1: out+=[(n//2)+1]*2
    return out[:2*n]
def motif_ladder(n:int)->Seq:
    out=[]; a,b=1,2
    while len(out)<2*n:
        out+=[a, b if b<=n else a]; a+=1; b+=1
        if a>n:a=n; 
        if b>n:b=n
    cnt={i:0 for i in range(1,n+1)}; fixed=[]
    for x in out:
        if cnt[x]<2: fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i]<2: fixed.append(i); cnt[i]+=1
    return fixed[:2*n]
def motif_corner_combo(n:int)->Tuple[Seq,Seq]: return motif_rainbow(n), motif_doubled(n)
def motif_seeds(n:int)->List[Seq]:
    S=[motif_rainbow(n), motif_doubled(n), motif_interleave(n), motif_zipper(n),
       motif_ladder(n), list(range(1,n+1))+list(range(1,n+1)),
       [x for pair in zip(range(1,n+1),range(1,n+1)) for x in pair]]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)): out.append(s)
    return out

# =========================
# Exact evaluator + duals
# =========================
Rect=Tuple[Tuple[int,int],Tuple[int,int]]; Point=Tuple[int,int]
def build_rects(H:Seq,V:Seq)->List[Rect]:
    X=seq_spans(H); Y=seq_spans(V); rects=[]
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2:x1,x2=x2,x1
        if y1>y2:y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects:List[Rect])->List[Point]:
    xs=sorted({x for r in rects for x in (r[0][0],r[0][1])})
    ys=sorted({y for r in rects for y in (r[1][0],r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects:List[Rect], pts:List[Point])->Tuple[List[List[int]],List[List[int]]]:
    covers=[]; rect_to_constr=[[] for _ in rects]
    for p_idx,(x,y) in enumerate(pts):
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1<=x<=x2) and (y1<=y<=y2): S.append(i)
        covers.append(S)
        for i in S: rect_to_constr[i].append(p_idx)
    return covers,rect_to_constr

@dataclass
class EvalInfo:
    lp:float; ilp:float; ratio:float
    covers:List[List[int]]; rect_to_constr:List[List[int]]
    duals:List[float]; dual_gain_per_rect:List[float]
    overlap_heavy_count:int; corner_load:int
    H_heat:List[float]; V_heat:List[float]

EVAL_CACHE:Dict[str,EvalInfo]={}

def _compute_corner_load(rects:List[Rect])->int:
    cnt={}
    for ((x1,x2),(y1,y2)) in rects:
        for pt in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]: cnt[pt]=cnt.get(pt,0)+1
    return sum(cnt.values())

def _project_heat_H(rects:List[Rect],gain:List[float],L:int)->List[float]:
    heat=[0.0]*L
    for i,((x1,x2),_) in enumerate(rects):
        if x2>=x1:
            w=gain[i]/max(1,(x2-x1+1))
            for x in range(x1,x2+1): heat[x]+=w
    return heat
def _project_heat_V(rects:List[Rect],gain:List[float],L:int)->List[float]:
    heat=[0.0]*L
    for i,(_, (y1,y2)) in enumerate(rects):
        if y2>=y1:
            w=gain[i]/max(1,(y2-y1+1))
            for y in range(y1,y2+1): heat[y]+=w
    return heat

def solve_lp_ilp_with_duals(rects:List[Rect], grb_threads:int=0)->Tuple[float,float,List[float],List[List[int]],List[List[int]]]:
    pts=grid_points(rects); covers,rect_to_constr=covers_grid_closed(rects,pts)
    # LP
    m_lp=gp.Model("misr_lp"); m_lp.setParam('OutputFlag',0)
    if grb_threads>0: m_lp.setParam('Threads',grb_threads)
    n=len(rects)
    x=m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    constr_handles=[]
    for S in covers:
        if S: c=m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1.0); constr_handles.append(c)
        else: constr_handles.append(None)
    m_lp.optimize()
    lp=float(m_lp.objVal) if m_lp.status==GRB.OPTIMAL else 0.0
    duals=[0.0]*len(covers)
    if m_lp.status==GRB.OPTIMAL:
        for idx,ch in enumerate(constr_handles):
            if ch is not None: duals[idx]=ch.Pi

    # ILP
    m_ilp=gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag',0)
    if grb_threads>0: m_ilp.setParam('Threads',grb_threads)
    y=m_ilp.addVars(n, vtype=GRB.BINARY)
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1.0)
    m_ilp.optimize()
    ilp=float(m_ilp.objVal) if m_ilp.status==GRB.OPTIMAL else 0.0
    return lp,ilp,duals,covers,rect_to_constr

def evaluate_instance(H:Seq,V:Seq, grb_threads:int=0)->EvalInfo:
    k=instance_key(H,V)
    if k in EVAL_CACHE: return EVAL_CACHE[k]
    rects=build_rects(H,V); Lh,Lv=len(H),len(V)
    lp,ilp,duals,covers,rect_to_constr=solve_lp_ilp_with_duals(rects,grb_threads)
    ratio=(lp/ilp) if ilp>0 else 0.0
    gain=[0.0]*len(rects)
    for i,cons in enumerate(rect_to_constr): gain[i]=sum(duals[p] for p in cons)
    cov_counts=[len(S) for S in covers]
    overlap_heavy=sum(1 for c in cov_counts if c>=3)
    corner_load=_compute_corner_load(rects)
    H_heat=_project_heat_H(rects,gain,Lh); V_heat=_project_heat_V(rects,gain,Lv)
    info=EvalInfo(lp,ilp,ratio,covers,rect_to_constr,duals,gain,overlap_heavy,corner_load,H_heat,V_heat)
    EVAL_CACHE[k]=info; return info

# =========================
# Blended score
# =========================
def blended_score(info:EvalInfo,n:int,bias_mode:str, a:float,b:float,g:float, lo:float,lc:float)->float:
    sc = info.ratio + a*(info.lp/max(1,n)) - b*(info.ilp/max(1,n))
    if bias_mode in ("dual","dual+corners"):
        gsig = float(np.mean(info.dual_gain_per_rect)) if info.dual_gain_per_rect else 0.0
        sc += g*gsig
    if bias_mode=="dual+corners":
        sc += lo*(info.overlap_heavy_count/max(1,n))
        sc += lc*(info.corner_load/max(1,n))
    return float(sc)

# =========================
# Diversity helpers
# =========================
def norm_hamm(a:Seq,b:Seq)->float:
    L=min(len(a),len(b))
    return sum(1 for i in range(L) if a[i]!=b[i]) / float(L if L>0 else 1)

def inst_distance(h1:Seq,v1:Seq,h2:Seq,v2:Seq)->float:
    return 0.5*(norm_hamm(h1,h2)+norm_hamm(v1,v2))

# =========================
# Neighbors (hot + generic)
# =========================
def guided_window(heat: List[float], rng: random.Random, L: int, span: int) -> Tuple[int, int]:
    """Pick a start index weighted by `heat`, then open a window of length ~span."""
    if L <= 1:
        return 0, 0
    if not heat or len(heat) != L:
        a = rng.randrange(L)
        b = min(L - 1, a + max(1, span))
        return (a, b) if a <= b else (b, a)

    w = np.asarray(heat, dtype=float)
    # make strictly positive; guard against nan/inf
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = w - w.min() + 1e-9
    s = float(w.sum())

    if s <= 0.0:
        i = rng.randrange(L)
    else:
        r = rng.random() * s
        c = 0.0
        i = 0
        for idx, wi in enumerate(w):
            c += float(wi)
            if r <= c:
                i = idx
                break

    j = min(L - 1, i + max(1, span))
    return (i, j) if i <= j else (j, i)

def neighbors(H:Seq,V:Seq, info:EvalInfo, rng:random.Random, k:int=140)->List[Instance]:
    out=[]; L=len(H)
    for _ in range(k):
        mode = rng.random()
        if mode < 0.4:
            # hot-window reverse on H or V
            on_H = rng.random()<0.5
            if on_H:
                a,b=guided_window(info.H_heat,rng,L, span=rng.randint(2,max(3,L//6)))
                A=H[:]; A[a:b+1]=reversed(A[a:b+1]); out.append(canonicalize(A,V))
            else:
                a,b=guided_window(info.V_heat,rng,L, span=rng.randint(2,max(3,L//6)))
                A=V[:]; A[a:b+1]=reversed(A[a:b+1]); out.append(canonicalize(H,A))
        elif mode < 0.7:
            # hot label pair swap (pick two hot spans)
            gh=np.array(info.H_heat if info.H_heat else [1.0]*L)
            idx=gh.argsort()[-max(2,L//4):]
            if len(idx)>=2:
                i,j=sorted(rng.sample(list(map(int,idx)),2))
                A=H[:]; A[i],A[j]=A[j],A[i]; out.append(canonicalize(A,V))
            else:
                A=H[:]; i,j=rng.randrange(L),rng.randrange(L); A[i],A[j]=A[j],A[i]; out.append(canonicalize(A,V))
        else:
            # generic block move or pair swap
            if rng.random()<0.5:
                a,b=sorted((rng.randrange(L),rng.randrange(L)))
                if a!=b:
                    A=H[:] if rng.random()<0.5 else V[:]
                    blk=A[a:b+1]; del A[a:b+1]; t=rng.randrange(len(A)+1); A[t:t]=blk
                    out.append(canonicalize(A,V) if len(A)==len(H) else canonicalize(H,A))
            else:
                labs=list(set(H)); 
                if len(labs)>=2:
                    a_lab,b_lab=rng.sample(labs,2)
                    for base in (0,1):
                        A=(H[:] if base==0 else V[:])
                        pa=[i for i,x in enumerate(A) if x==a_lab]; pb=[i for i,x in enumerate(A) if x==b_lab]
                        if len(pa)==2 and len(pb)==2:
                            for ia,ib in zip(pa,pb): A[ia],A[ib]=A[ib],A[ia]
                            out.append(canonicalize(A,V) if base==0 else canonicalize(H,A))
                else:
                    A=H[:]; i,j=rng.randrange(L),rng.randrange(L); A[i],A[j]=A[j],A[i]; out.append(canonicalize(A,V))
    return out

# =========================
# Local search (tabu + SA)
# =========================
def local_search(seed:Instance, time_budget_s:float, rng:random.Random,
                 bias_mode:str, a:float,b:float,g:float, lo:float,lc:float,
                 grb_threads:int=0,
                 elite_size:int=64, neighbor_k:int=140,
                 sa_T0:float=0.08,
                 stagnation_shakes:int=2):
    start=time.time()
    H,V=canonicalize(*seed); n=max(H) if H else 1
    elites:List[Tuple[float,Seq,Seq]]=[]; best=-1.0
    info=evaluate_instance(H,V,grb_threads); blended0=blended_score(info,n,bias_mode,a,b,g,lo,lc)
    def push(score,h,v):
        elites.append((score,h[:],v[:])); elites.sort(key=lambda x:-x[0])
        if len(elites)>elite_size: elites.pop()
    push(info.ratio,H,V); best=max(best,info.ratio)

    last_improve=time.time()
    while time.time()-start < time_budget_s:
        cand=neighbors(H,V,info,rng,neighbor_k)
        best_nb=None; best_sc=-1e18
        for (h2,v2) in cand:
            inf2=evaluate_instance(h2,v2,grb_threads)
            sc2=blended_score(inf2,n,bias_mode,a,b,g,lo,lc)
            push(inf2.ratio,h2,v2)
            if sc2>best_sc: best_sc=sc2; best_nb=(h2,v2,inf2,sc2)
        if best_nb:
            h2,v2,inf2,sc2=best_nb
            if sc2>=blended0 or math.exp((sc2-blended0)/max(1e-6,sa_T0))>rng.random():
                H,V,info,blended0=h2,v2,inf2,sc2
                if info.ratio>best+1e-12: best=info.ratio; last_improve=time.time()
        # shake if stagnant
        if time.time()-last_improve>max(3.0, 0.3*time_budget_s) and stagnation_shakes>0:
            stagnation_shakes-=1; last_improve=time.time()
            # heavy shake on hottest window
            L=len(H); a1,b1=guided_window(info.H_heat,rng,L, span=max(3,L//4))
            A=H[:]; A[a1:b1+1]=reversed(A[a1:b1+1])
            a2,b2=guided_window(info.V_heat,rng,L, span=max(3,L//4))
            B=V[:]; B[a2:b2+1]=reversed(B[a2:b2+1])
            H,V=canonicalize(A,B); info=evaluate_instance(H,V,grb_threads)
            blended0=blended_score(info,n,bias_mode,a,b,g,lo,lc)
    elites_sorted=sorted(elites,key=lambda x:-x[0])
    return elites_sorted,best

# =========================
# Tiny Transformer proposer
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,max_len:int=4096):
        super().__init__()
        pe=torch.zeros(max_len,d_model); pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe)
    def forward(self,x): return x + self.pe[:x.size(1)]

class TinyGPT(nn.Module):
    def __init__(self,d=192,nhead=6,nlayers=3,dropout=0.1):
        super().__init__()
        self.label_embed=nn.Embedding(BASE_VOCAB+MAX_N,d)
        self.n_embed=nn.Embedding(MAX_N+1,d)
        self.pos=PositionalEncoding(d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=4*d,dropout=dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(layer,num_layers=nlayers)
        self.out=nn.Linear(d,BASE_VOCAB+MAX_N)
    def forward(self,tokens,n_scalar):
        tok=self.label_embed(tokens); nemb=self.n_embed(n_scalar).unsqueeze(1).expand_as(tok)
        x=self.pos(tok+nemb); L=x.size(1)
        mask=nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h=self.enc(x,mask=mask); return self.out(h)

def seq_to_tokens(seq:Seq)->List[int]: return [BASE_VOCAB+(i-1) for i in seq]
def tokens_to_seq(tokens:List[int])->Seq: return [t-BASE_VOCAB+1 for t in tokens]

@dataclass
class Batch:
    tokens:torch.Tensor; n_scalar:torch.Tensor; targets:torch.Tensor

def make_batch(elites:List[Tuple[float,Seq,Seq]], B:int, rng:random.Random)->Batch:
    tlist=[]; tglist=[]; ns=[]
    for _ in range(B):
        _,H,V=rng.choice(elites); n=max(H)
        tok=[SPECIAL["BOS"]]+seq_to_tokens(H)+[SPECIAL["SEP"]]+seq_to_tokens(V)+[SPECIAL["EOS"]]
        tgt=tok[1:]+[SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok)); tglist.append(torch.tensor(tgt)); ns.append(n)
    L=max(len(t) for t in tlist); pad=SPECIAL["EOS"]
    tokens=torch.full((B,L),pad,dtype=torch.long); targets=torch.full((B,L),pad,dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)]=t; targets[i,:len(tt)]=tt
    return Batch(tokens.to(DEVICE), torch.tensor(ns,dtype=torch.long,device=DEVICE), targets.to(DEVICE))

@torch.no_grad()
def sample_model(model:TinyGPT,n:int,temperature:float=1.0,top_p:float=0.9,max_len:int=4096)->Instance:
    model.eval(); toks=[SPECIAL["BOS"]]
    def step(mask_valid:List[bool])->int:
        inp=torch.tensor(toks,dtype=torch.long,device=DEVICE).unsqueeze(0)
        nvec=torch.tensor([n],dtype=torch.long,device=DEVICE)
        logits=model(inp,nvec)[0,-1]
        mask=torch.tensor(mask_valid,device=DEVICE); logits=logits.masked_fill(~mask,-1e9)
        probs=F.softmax(logits/temperature,dim=-1); sorted_probs,idx=torch.sort(probs,descending=True)
        csum=torch.cumsum(sorted_probs,dim=-1); keep=csum<=top_p
        if not torch.any(keep): keep[0]=True
        p=torch.zeros_like(probs).scatter(0, idx[keep], sorted_probs[keep]); p=p/p.sum()
        return int(torch.multinomial(p,1).item())
    counts=[0]*(n+1)
    while sum(counts)<2*n:
        mask=[False]*(BASE_VOCAB+MAX_N)
        for i in range(1,n+1):
            if counts[i]<2: mask[BASE_VOCAB+(i-1)]=True
        toks.append(step(mask)); lab=toks[-1]-BASE_VOCAB+1; counts[lab]+=1
    toks.append(SPECIAL["SEP"])
    counts=[0]*(n+1)
    while sum(counts)<2*n and len(toks)<max_len:
        mask=[False]*(BASE_VOCAB+MAX_N)
        for i in range(1,n+1):
            if counts[i]<2: mask[BASE_VOCAB+(i-1)]=True
        toks.append(step(mask)); lab=toks[-1]-BASE_VOCAB+1; counts[lab]+=1
    sep=toks.index(SPECIAL["SEP"]); H=tokens_to_seq(toks[1:sep]); V=tokens_to_seq(toks[sep+1:])
    return canonicalize(H,V)

def train_one_step(model:nn.Module,opt:torch.optim.Optimizer,batch:Batch)->float:
    model.train(); logits=model(batch.tokens,batch.n_scalar)
    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), batch.targets.reshape(-1), ignore_index=SPECIAL["EOS"])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return float(loss.item())

# =========================
# Save helpers
# =========================
def atomic_save(obj, path:str):
    import pickle
    d=os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".tmp_",dir=d); os.close(fd)
    try:
        with open(tmp,"wb") as f: pickle.dump(obj,f)
        os.replace(tmp,path)
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception: pass

# =========================
# Seeding & lifting
# =========================
def seeded_pool(n:int,rng:random.Random,count:int,policy:str)->List[Instance]:
    if policy=="random": return [(random_valid_seq(n,rng),random_valid_seq(n,rng)) for _ in range(count)]
    motifs=motif_seeds(n); seeds=[]
    for m in motifs:
        seeds.append((m[:],random_valid_seq(n,rng))); seeds.append((random_valid_seq(n,rng),m[:]))
    RH,RV=motif_corner_combo(n); seeds+=[(RH[:],RV[:]),(RV[:],RH[:])]
    for i in range(min(len(motifs)-1,5 if policy=="mixed" else 7)):
        seeds.append((motifs[i][:],motifs[i+1][:]))
    while len(seeds)<count:
        if policy=="motif":
            a=rng.choice(motifs); b=rng.choice(motifs); seeds.append((a[:],b[:]))
        else:
            seeds.append((random_valid_seq(n,rng),random_valid_seq(n,rng)))
    return seeds[:count]

def recombine(elites:List[Tuple[float,Seq,Seq]], k:int, rng:random.Random)->List[Instance]:
    out=[]; pool=elites[:max(k*2,2)]
    for _ in range(k if pool else 0):
        _,H1,_=rng.choice(pool); _,_,V2=rng.choice(pool)
        out.append(canonicalize(H1,V2))
    return out

def choose_idx_by_weights(w:List[float], rng:random.Random)->int:
    if not w: return 0
    arr=np.array(w,dtype=float); arr=arr-arr.min()+1e-8; s=arr.sum()
    if s<=0: return rng.randrange(len(w))
    r=rng.random()*s; c=0.0
    for i,val in enumerate(arr):
        c+=float(val); 
        if r<=c: return i
    return len(w)-1

def lift_dualaware(H:Seq,V:Seq,n_new:int,rng:random.Random,H_heat:Optional[List[float]],V_heat:Optional[List[float]])->Instance:
    assert n_new>=max(H); H2,Hh=H[:],(H_heat if H_heat and len(H_heat)==len(H) else None)
    V2,Vh=V[:],(V_heat if V_heat and len(V_heat)==len(V) else None)
    for lab in range(max(H)+1,n_new+1):
        for (seq,heat) in ((H2,Hh),(V2,Vh)):
            if heat is None:
                # fallback depth measure
                sp=seq_spans(seq); line=[0]*(len(seq)+1)
                for (l,r) in sp:
                    if l<r:
                        line[l]+=1; 
                        if r+1<len(line): line[r+1]-=1
                d=[]; cur=0
                for i in range(len(seq)): cur+=line[i]; d.append(cur)
                i=choose_idx_by_weights(d,rng); j=min(i+1,len(seq))
            else:
                i=choose_idx_by_weights(heat,rng); j=min(i+1,len(seq))
            seq.insert(i,lab); seq.insert(j,lab)
    return canonicalize(H2,V2)

# =========================
# PatternBoost runner
# =========================
def run(
    seed:int=123,
    n_start:int=8, n_target:int=32,
    rounds_per_n:int=10,
    seeds_per_round:int=64,
    local_time_per_seed:float=4.0,
    elites_to_train:int=128, batch_size:int=32, train_steps_per_round:int=60,
    temperature:float=1.0, top_p:float=0.9,
    # bias (hi -> lo within each n via anneal)
    bias_mode:str="dual+corners",
    alpha_hi:float=0.45, alpha_lo:float=0.05,
    beta_hi:float=0.08,  beta_lo:float=0.22,
    gamma_hi:float=0.20, gamma_lo:float=0.00,
    lambda_overlap_hi:float=0.12, lambda_overlap_lo:float=0.00,
    lambda_corner_hi:float=0.08, lambda_corner_lo:float=0.00,
    phase_split:float=0.6,          # first frac exploit bias; then fade to ratio
    # search internals
    grb_threads:int=0, lift_step:int=3, seed_policy:str="mixed",
    neighbor_k:int=140, sa_T0:float=0.08,
    # diversity
    novelty_tau:float=0.12,         # min avg(H,V) normalized Hamming to admit if score similar
    novelty_soft_margin:float=0.01, # allow close clones only if strictly better by this
):
    rng=random.Random(seed); torch.manual_seed(seed)
    model=TinyGPT(d=192,nhead=6,nlayers=3).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    elites:List[Tuple[float,Seq,Seq]]=[]
    def diverse_push(score:float,H:Seq,V:Seq):
        # keep sorted by ratio, but filter near-duplicates
        nonlocal elites
        if not elites: elites=[(score,H[:],V[:])]; return
        # quick gate vs top 128
        base=elites[:min(128,len(elites))]
        max_sim = max(1.0-inst_distance(H,V,eh,ev) for _,eh,ev in base)
        # admit if sufficiently different OR strictly higher than current tail
        if (1.0-max_sim) >= novelty_tau or score >= elites[-1][0] + novelty_soft_margin:
            elites.append((score,H[:],V[:])); elites.sort(key=lambda x:-x[0])
            if len(elites)>4096: elites=elites[:4096]

    n=n_start; best_overall=0.0
    seeds=seeded_pool(n,rng,seeds_per_round,policy=seed_policy)
    round_global=0

    while n<=n_target:
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")
        for r in range(1,rounds_per_n+1):
            frac = (r-1)/max(1,rounds_per_n-1)
            # two-phase: bias → ratio
            blend = min(1.0, max(0.0, (frac - phase_split) / max(1e-9, 1.0-phase_split)))
            a = alpha_hi*(1-blend) + alpha_lo*blend
            b = beta_hi *(1-blend) + beta_lo *blend
            g = gamma_hi*(1-blend) + gamma_lo*blend
            lo= lambda_overlap_hi*(1-blend) + lambda_overlap_lo*blend
            lc= lambda_corner_hi *(1-blend) + lambda_corner_lo *blend

            # 1) local search
            per_seed_best=[]
            for (H,V) in seeds:
                es,best = local_search((H,V),
                                       time_budget_s=local_time_per_seed,
                                       rng=rng,
                                       bias_mode=bias_mode, a=a,b=b,g=g, lo=lo,lc=lc,
                                       grb_threads=grb_threads,
                                       elite_size=64, neighbor_k=neighbor_k,
                                       sa_T0=sa_T0, stagnation_shakes=2)
                for (score,h,v) in es: diverse_push(score,h,v)
                if best is not None: best_overall=max(best_overall,best)
                per_seed_best.append(best if best is not None else 0.0)
            med=np.median([x for x in per_seed_best if x>0]) if per_seed_best else 0.0
            # stats of current elites slice
            cur_ratios=[sc for (sc,_,_) in elites[:min(256,len(elites))]]
            if cur_ratios:
                q50=float(np.percentile(cur_ratios,50)); q75=float(np.percentile(cur_ratios,75))
                q90=float(np.percentile(cur_ratios,90)); q99=float(np.percentile(cur_ratios,99))
                print(f"[round {r}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f} med_seedbest={med:.4f} | q50={q50:.3f} q75={q75:.3f} q90={q90:.3f} q99={q99:.3f}")
            else:
                print(f"[round {r}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f} med_seedbest={med:.4f}")

            # 2) train proposer
            topk=elites[:max(elites_to_train, min(32,len(elites)))]
            if topk:
                last_loss=None
                for _ in range(train_steps_per_round):
                    batch=make_batch(topk, min(batch_size,len(topk)), rng)
                    last_loss=train_one_step(model,opt,batch)
                print(f"   trained {train_steps_per_round} steps, last loss ~ {last_loss:.3f}")
            # 2.5) checkpoint
            atomic_save(elites[:256], f"misr_elites_n{n}_r{r}.pkl")

            # 3) new seeds (recombine + model + jitter) with diversity
            new_seeds=recombine(elites[:128], k=max(1,seeds_per_round//4), rng=rng)
            def maybe_add(h,v):
                # novelty gate vs a small elite slice
                if elites:
                    base=elites[:min(96,len(elites))]
                    sim=max(1.0-inst_distance(h,v,eh,ev) for _,eh,ev in base)
                    if (1.0-sim) < novelty_tau*0.75 and rng.random()<0.7:
                        return
                new_seeds.append((h,v))
            while len(new_seeds)<seeds_per_round:
                if elites and rng.random()<0.28:
                    _,h,v=rng.choice(elites[:64]); h=h[:]; v=v[:]
                    # strong jitter on hot windows (pseudo)
                    a1,b1 = 0, max(1,len(h)//3); 
                    if rng.random()<0.7:
                        a1,b1 = rng.randrange(len(h)), rng.randrange(len(h))
                        if a1>b1: a1,b1=b1,a1
                    h[a1:b1+1]=reversed(h[a1:b1+1])
                    a2,b2 = 0, max(1,len(v)//3)
                    if rng.random()<0.7:
                        a2,b2 = rng.randrange(len(v)), rng.randrange(len(v))
                        if a2>b2:a2,b2=b2,a2
                    v[a2:b2+1]=reversed(v[a2:b2+1])
                    maybe_add(*canonicalize(h,v))
                else:
                    h,v=sample_model(model,n,temperature=temperature,top_p=top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        h,v=random_valid_seq(n,rng),random_valid_seq(n,rng)
                    # random reverse block
                    if rng.random()<0.35:
                        for S in (h,v):
                            a,b=sorted((rng.randrange(len(S)),rng.randrange(len(S))))
                            if a!=b: S[a:b+1]=reversed(S[a:b+1])
                    maybe_add(*canonicalize(h,v))

            # optional motif refresh
            if seed_policy!="random":
                RH,RV=motif_corner_combo(n); mix=[(RH[:],RV[:]),(RV[:],RH[:])]
                for m in motif_seeds(n)[:2]: mix.append((m[:], motif_rainbow(n)))
                for i in range(min(len(mix), max(2,seeds_per_round//10))): new_seeds[i]=mix[i]
            seeds=new_seeds

        # Save per-n final
        atomic_save(elites[:256], f"misr_elites_n{n}_final.pkl")

        # 4) lift to next n
        if elites:
            n_next=n+lift_step
            lifted=[]
            top_for_lift=elites[:min(120,len(elites))]
            for _,h,v in top_for_lift:
                inf=EVAL_CACHE.get(instance_key(h,v))
                h2,v2=lift_dualaware(h,v,n_next,rng, inf.H_heat if inf else None, inf.V_heat if inf else None)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round-len(lifted))]
            n=n_next
        else:
            seeds=seeded_pool(n,rng,seeds_per_round,policy=seed_policy)

    print("\n=== BEST ELITES (top 10) ===")
    for i,(sc,h,v) in enumerate(elites[:10]): print(f"#{i+1} ratio={sc:.4f} n={max(h)}")
    return elites

# =========================
# CLI
# =========================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=64)
    ap.add_argument("--local_time_per_seed", type=float, default=4.0)
    ap.add_argument("--elites_to_train", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--bias_mode", type=str, default="dual+corners", choices=["simple","dual","dual+corners"])
    ap.add_argument("--alpha_hi", type=float, default=0.45)
    ap.add_argument("--alpha_lo", type=float, default=0.05)
    ap.add_argument("--beta_hi",  type=float, default=0.08)
    ap.add_argument("--beta_lo",  type=float, default=0.22)
    ap.add_argument("--gamma_hi", type=float, default=0.20)
    ap.add_argument("--gamma_lo", type=float, default=0.00)
    ap.add_argument("--lambda_overlap_hi", type=float, default=0.12)
    ap.add_argument("--lambda_overlap_lo", type=float, default=0.00)
    ap.add_argument("--lambda_corner_hi", type=float, default=0.08)
    ap.add_argument("--lambda_corner_lo", type=float, default=0.00)
    ap.add_argument("--phase_split", type=float, default=0.6)

    ap.add_argument("--grb_threads", type=int, default=0)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--seed_policy", type=str, default="mixed", choices=["mixed","random","motif"])
    ap.add_argument("--neighbor_k", type=int, default=140)
    ap.add_argument("--sa_T0", type=float, default=0.08)
    ap.add_argument("--novelty_tau", type=float, default=0.12)
    ap.add_argument("--novelty_soft_margin", type=float, default=0.01)

    args=ap.parse_args()
    print(f"Device: {DEVICE}")
    elites=[]
    try:
        elites=run(seed=args.seed,
                   n_start=args.n_start, n_target=args.n_target,
                   rounds_per_n=args.rounds_per_n,
                   seeds_per_round=args.seeds_per_round,
                   local_time_per_seed=args.local_time_per_seed,
                   elites_to_train=args.elites_to_train, batch_size=args.batch_size,
                   train_steps_per_round=args.train_steps_per_round,
                   temperature=args.temperature, top_p=args.top_p,
                   bias_mode=args.bias_mode,
                   alpha_hi=args.alpha_hi, alpha_lo=args.alpha_lo,
                   beta_hi=args.beta_hi,  beta_lo=args.beta_lo,
                   gamma_hi=args.gamma_hi, gamma_lo=args.gamma_lo,
                   lambda_overlap_hi=args.lambda_overlap_hi, lambda_overlap_lo=args.lambda_overlap_lo,
                   lambda_corner_hi=args.lambda_corner_hi, lambda_corner_lo=args.lambda_corner_lo,
                   phase_split=args.phase_split,
                   grb_threads=args.grb_threads, lift_step=args.lift_step,
                   seed_policy=args.seed_policy, neighbor_k=args.neighbor_k, sa_T0=args.sa_T0,
                   novelty_tau=args.novelty_tau, novelty_soft_margin=args.novelty_soft_margin)
    finally:
        try:
            atomic_save((elites or [])[:256], "misr_elites.pkl")
            print("Saved top elites to misr_elites.pkl")
        except Exception as e:
            print(f"WARNING: failed to save final elites: {e}", file=sys.stderr)

if __name__=="__main__":
    main()
