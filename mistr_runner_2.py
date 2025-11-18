# mistr_runner_deep.py
from __future__ import annotations
import argparse, json, math, os, pickle, random, time, warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB

def _maybe_import_chuzhoy():
    try:
        from src.chuzhoy_seeds import seeds_for_n_from_chuzhoy
        return seeds_for_n_from_chuzhoy
    except Exception:
        return None
seeds_for_n_from_chuzhoy = _maybe_import_chuzhoy()

warnings.filterwarnings("ignore", message=r".*TF32 behavior.*", category=UserWarning)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]

SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3
MAX_N = 128

# ---------------- utils & motifs ----------------
def load_seeds_from_pkl(seed_pkl: str, n: int, top_k: int = 10) -> List[Instance]:
    try:
        with open(seed_pkl, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return []
    bag = []
    for item in data:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3 and isinstance(item[1], list) and isinstance(item[2], list):
                ratio, H, V = item[:3]
                if H and max(H) == n:
                    bag.append((float(ratio), H, V))
            elif len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
                H, V = item
                if H and max(H) == n:
                    bag.append((0.0, H, V))
    bag.sort(key=lambda t: -t[0])
    return [(H[:], V[:]) for _, H, V in bag[:top_k]]

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    seq = [i for i in range(1, n + 1) for _ in range(2)]
    rng.shuffle(seq)
    return seq

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    first = {}
    spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def canonicalize(H: Seq, V: Seq) -> Instance:
    order, seen = [], set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def instance_key(H: Seq, V: Seq) -> str:
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    import hashlib
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    out=[]
    for i in range(1, n+1):
        out += [i,i]
    return out

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_zipper(n: int) -> Seq:
    out=[]
    for i in range(1, (n//2)+1):
        j = n - i + 1
        out += [i, j, i, j]
    if n % 2 == 1:
        k = (n//2)+1
        out += [k,k]
    return out[:2*n]

def motif_ladder(n: int) -> Seq:
    out=[]; a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1; b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_corner_combo(n: int) -> Tuple[Seq, Seq]:
    return motif_rainbow(n), motif_doubled(n)

def motif_seeds(n: int) -> List[Seq]:
    S = [
        motif_rainbow(n), motif_doubled(n), motif_interleave(n),
        motif_zipper(n),  motif_ladder(n),
        list(range(1, n+1)) + list(range(1, n+1)),
        [x for pair in zip(range(1,n+1), range(1,n+1)) for x in pair],
    ]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)):
            out.append(s)
    return out

def lift_instance(H: Seq, V: Seq, n_new: int, rng: random.Random) -> Instance:
    assert n_new >= max(H)
    H2, V2 = H[:], V[:]
    def depth(seq: Seq) -> List[int]:
        sp = seq_spans(seq)
        line = [0]*(len(seq)+1)
        for (l,r) in sp:
            if l < r:
                line[l] += 1
                if r+1 < len(line):
                    line[r+1] -= 1
        out=[]; cur=0
        for i in range(len(seq)):
            cur += line[i]; out.append(cur)
        return out
    def weighted_idx(weights: List[int]) -> int:
        tot = sum(w+1 for w in weights)
        x = rng.randrange(tot); s=0
        for i,w in enumerate(weights):
            s += (w+1)
            if x < s: return i
        return len(weights)-1
    for lab in range(max(H)+1, n_new+1):
        for seq in (H2, V2):
            d = depth(seq)
            i = weighted_idx(d); j = min(i+1, len(seq))
            seq.insert(i, lab); seq.insert(j, lab)
    return canonicalize(H2, V2)

# ------------- rects & scoring -------------
def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects: List[Rect]) -> List[Tuple[int,int]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Tuple[int,int]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

def solve_lp_ilp(rects: List[Rect], grb_threads: int = 0) -> Tuple[float, float]:
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    return lp, ilp

def score_ratio(H: Seq, V: Seq, alpha_lp=0.0, beta_ilp=0.0, grb_threads=0):
    rects = build_rects(H,V)
    lp, ilp = solve_lp_ilp(rects, grb_threads=grb_threads)
    ratio = (lp/ilp) if ilp > 0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha_lp * (lp / n) - beta_ilp * (ilp / n)
    return lp, ilp, ratio, blended

# ------------- model (BetterGPT) + aux adjacency -------------
class BetterGPT(nn.Module):
    def __init__(self, d=384, nhead=8, nlayers=8, ff_mult=4, dropout=0.1, max_len=4096):
        super().__init__()
        vocab = BASE_VOCAB + MAX_N
        self.label_embed = nn.Embedding(vocab, d)
        self.n_embed     = nn.Embedding(MAX_N + 1, d)
        self.pos_embed   = nn.Embedding(max_len, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=ff_mult * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out = nn.Linear(d, vocab)

    def forward(self, tokens: torch.Tensor, n_scalar: torch.Tensor, return_hidden: bool = False):
        B, L = tokens.size()
        device = tokens.device
        tok_emb = self.label_embed(tokens)                         # [B, L, d]
        n_emb  = self.n_embed(n_scalar).unsqueeze(1).expand(-1, L, -1)  # [B, L, d]
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_idx)                          # [B, L, d]
        x = tok_emb + n_emb + pos_emb
        mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        h = self.enc(x, mask=mask)        # [B, L, d]
        logits = self.out(h)              # [B, L, V]
        if return_hidden:
            return logits, h
        return logits

def seq_to_tokens(seq: Seq) -> List[int]:
    return [BASE_VOCAB + (i-1) for i in seq]

def tokens_to_seq(tokens: List[int]) -> Seq:
    return [t - BASE_VOCAB + 1 for t in tokens]

@dataclass
class Batch:
    tokens: torch.Tensor
    n_scalar: torch.Tensor
    targets: torch.Tensor

def make_batch(elites: List[Tuple[float, Seq, Seq]], B: int, rng: random.Random) -> Batch:
    tlist=[]; tglist=[]; ns=[]
    for _ in range(B):
        _, H, V = rng.choice(elites)
        n = max(H)
        tok = [SPECIAL["BOS"]] + seq_to_tokens(H) + [SPECIAL["SEP"]] + seq_to_tokens(V) + [SPECIAL["EOS"]]
        tgt = tok[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
        ns.append(n)
    L = max(len(t) for t in tlist)
    pad = SPECIAL["EOS"]
    tokens = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)]  = t
        targets[i,:len(tt)] = tt
    return Batch(tokens=tokens.to(DEVICE),
                 n_scalar=torch.tensor(ns, dtype=torch.long, device=DEVICE),
                 targets=targets.to(DEVICE))

# ---- aux helpers ----
def _tokens_to_HV(tokens_1d: List[int]) -> Tuple[List[int], List[int]]:
    try:
        sep = tokens_1d.index(SPECIAL["SEP"])
    except ValueError:
        sep = len(tokens_1d)//2
    H_t = tokens_1d[1:sep]                     # drop BOS
    V_t = tokens_1d[sep+1:]                    # drop SEP
    H = [t - BASE_VOCAB + 1 for t in H_t if t >= BASE_VOCAB]
    V = [t - BASE_VOCAB + 1 for t in V_t if t >= BASE_VOCAB]
    return H, V

def _adjacency_from_HV(H: List[int], V: List[int], n: int) -> np.ndarray:
    def spans(seq):
        first = {}
        out = [None]*(n+1)
        for k, lab in enumerate(seq):
            if lab not in first: first[lab] = k
            else: out[lab] = (first[lab], k)
        return out
    X = spans(H); Y = spans(V)
    def overlap(a, b):
        (l1, r1), (l2, r2) = a, b
        if l1 > r1: l1, r1 = r1, l1
        if l2 > r2: l2, r2 = r2, l2
        return not (r1 < l2 or r2 < l1)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n+1):
        if X[i] is None or Y[i] is None: continue
        for j in range(i+1, n+1):
            if X[j] is None or Y[j] is None: continue
            if overlap(X[i], X[j]) and overlap(Y[i], Y[j]):
                A[i-1, j-1] = 1.0
                A[j-1, i-1] = 1.0
    return A

def _pool_label_embeddings(h: torch.Tensor, tokens: torch.Tensor, n: int) -> torch.Tensor:
    B, L, d = h.size()
    dev = h.device
    label_ids = tokens - BASE_VOCAB + 1
    is_label  = (tokens >= BASE_VOCAB) & (tokens != SPECIAL["EOS"])
    pooled = torch.zeros(B, n, d, device=dev)
    counts = torch.zeros(B, n, 1, device=dev)
    for b in range(B):
        ids_b = label_ids[b]
        mask_b = is_label[b]
        h_b = h[b]
        ids_valid = ids_b[mask_b]
        h_valid   = h_b[mask_b]
        if ids_valid.numel() > 0:
            idx = (ids_valid.clamp(min=1, max=n) - 1).long()
            pooled[b].index_add_(0, idx, h_valid)
            cnt = torch.ones_like(ids_valid, dtype=h_valid.dtype, device=dev).unsqueeze(1)
            counts[b].index_add_(0, idx, cnt)
    pooled = pooled / counts.clamp_min(1.0)
    return pooled

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch,
                   aux_adj: bool = True, aux_lambda: float = 0.25):
    model.train()
    logits, h = model(batch.tokens, batch.n_scalar, return_hidden=True)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.targets.reshape(-1),
        ignore_index=SPECIAL["EOS"]
    )
    if not aux_adj:
        loss = ce
    else:
        B, L = batch.tokens.size()
        n_list = batch.n_scalar.tolist()
        targets = []
        with torch.no_grad():
            tok_cpu = batch.tokens.detach().cpu().tolist()
            for b in range(B):
                n = n_list[b]
                H, V = _tokens_to_HV(tok_cpu[b])
                A = _adjacency_from_HV(H, V, n)
                iu, ju = np.triu_indices(n, k=1)
                targets.append(torch.tensor(A[iu, ju], dtype=torch.float32, device=h.device))
        pooled = _pool_label_embeddings(h, batch.tokens, max(n_list))
        vecs = []
        for b, n in enumerate(n_list):
            vecs.append(pooled[b, :n])
        E = torch.cat(vecs, dim=0)      # [sum_n, d]
        E = F.normalize(E, dim=-1)
        pair_logits = []
        offset = 0
        for b, n in enumerate(n_list):
            Eb = E[offset:offset+n]
            offset += n
            S = (Eb @ Eb.T) * (h.size(-1) ** 0.5)
            iu, ju = torch.triu_indices(n, n, offset=1, device=S.device)
            pair_logits.append(S[iu, ju])
        pair_logits = torch.cat(pair_logits, dim=0)
        target_vec = torch.cat(targets, dim=0)
        bce = F.binary_cross_entropy_with_logits(pair_logits, target_vec)
        loss = ce + aux_lambda * bce

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(ce.item())

# ------------- sampling -------------
@torch.no_grad()
def sample_model(model: BetterGPT, n: int, temperature: float = 1.0, top_p: float = 0.9, max_len: int = 4096) -> Instance:
    model.eval()
    toks = [SPECIAL["BOS"]]
    def step(mask_valid: List[bool]) -> int:
        inp = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
        nvec = torch.tensor([n], dtype=torch.long, device=DEVICE)
        logits = model(inp, nvec)[0, -1]
        mask = torch.tensor(mask_valid, device=DEVICE)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits/temperature, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        if not torch.any(keep): keep[0] = True
        p = torch.zeros_like(probs).scatter(0, idx[keep], sorted_probs[keep])
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())

    counts = [0]*(n+1)
    while sum(counts) < 2*n:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    toks.append(SPECIAL["SEP"])
    counts = [0]*(n+1)
    while sum(counts) < 2*n and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    sep_idx = toks.index(SPECIAL["SEP"])
    H_tok = toks[1:sep_idx]
    V_tok = toks[sep_idx+1:]
    H = tokens_to_seq(H_tok); V = tokens_to_seq(V_tok)
    return canonicalize(H,V)

# ------------- local search -------------
def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 96) -> List[Instance]:
    out=[]
    L = len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','pairHV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i], A[j] = A[j], A[i]
        elif which.startswith('move'):
            if i!=j:
                x = A.pop(i); A.insert(j, x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk = A[a:b+1]; del A[a:b+1]
                t = rng.randrange(len(A)+1); A[t:t]=blk
        elif which == 'pairHV':
            labs = list(set(H) & set(V))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                AH, AV = H[:], V[:]
                for S in (AH,):
                    pa = [i for i,x in enumerate(S) if x==a_lab]
                    pb = [i for i,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                for S in (AV,):
                    pa = [i for i,x in enumerate(S) if x==a_lab]
                    pb = [i for i,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                out.append(canonicalize(AH, AV)); continue
        elif which.startswith('rev'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                A[a:b+1] = list(reversed(A[a:b+1]))
        else:
            labs = list(set(A))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                pa = [idx for idx,x in enumerate(A) if x==a_lab]
                pb = [idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia, ib in zip(pa, pb):
                        A[ia], A[ib] = A[ib], A[ia]
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance, time_budget_s: float, rng: random.Random,
                 alpha_lp: float, beta_ilp: float, grb_threads: int = 0,
                 tabu_seconds: float = 20.0, elite_size: int = 64, neighbor_k: int = 96):
    start = time.time()
    H, V = canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]] = {}
    elites: List[Tuple[float, Seq, Seq]] = []
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, h: Seq, v: Seq):
        elites.append((score, h[:], v[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    while time.time() - start < time_budget_s:
        key = instance_key(H,V); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, H, V = random.choice(elites)
            else: H, V = H[::-1], V[::-1]
            continue

        if key not in seen:
            lp, ilp, ratio, blended = score_ratio(H,V, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
            seen[key] = (lp, ilp, ratio, blended)
            push(ratio, H, V)
            best = max(best, ratio)
        else:
            lp, ilp, ratio, blended = seen[key]

        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_sc=-1e9
        for (h2,v2) in cand:
            k2 = instance_key(h2,v2)
            if k2 in seen:
                lp2, ilp2, r2, b2 = seen[k2]
            else:
                lp2, ilp2, r2, b2 = score_ratio(h2,v2, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
                seen[k2] = (lp2, ilp2, r2, b2)
                push(r2, h2, v2)
            if b2 > best_sc:
                best_sc = b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)

        if best_nb:
            _,_,lp2,ilp2,r2,b2 = best_nb
            if b2 >= blended:
                H, V = best_nb[0], best_nb[1]
            else:
                delta = b2 - blended
                T = 0.03
                if math.exp(delta/max(T,1e-6)) > random.random():
                    H, V = best_nb[0], best_nb[1]
                else:
                    tabu[key] = now
                    if elites: _, H, V = random.choice(elites)
                    else: H, V = H[::-1], V[::-1]
            best = max(best, r2)

    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

# ------------- Ryan 1.5-gap seeds (n=18) -------------
_RYAN_18 = [
([2, 6, 7, 8, 9, 8, 10, 9, 1, 1, 15, 18, 7, 3, 4, 3, 14, 17, 2, 16, 15, 17, 16, 11, 10, 12, 13, 12, 13, 5, 4, 6, 11, 14, 5, 18],
 [1, 5, 6, 9, 12, 6, 8, 10, 9, 11, 10, 3, 11, 13, 12, 14, 15, 14, 16, 15, 4, 4, 5, 13, 2, 17, 2, 7, 8, 1, 16, 18, 3, 7, 17, 18]),
([10, 18, 11, 7, 12, 13, 2, 12, 3, 11, 14, 15, 14, 4, 3, 1, 6, 1, 9, 10, 2, 8, 7, 8, 9, 13, 5, 5, 17, 6, 16, 16, 17, 4, 15, 18],
 [5, 12, 14, 15, 16, 15, 4, 13, 13, 3, 14, 4, 1, 8, 17, 6, 6, 2, 9, 2, 16, 11, 12, 3, 10, 11, 10, 9, 7, 1, 8, 18, 5, 7, 17, 18]),
([2, 8, 9, 10, 9, 4, 11, 10, 3, 3, 14, 18, 1, 2, 6, 5, 17, 6, 12, 11, 15, 16, 5, 15, 13, 12, 13, 16, 17, 4, 14, 7, 7, 8, 18, 1],
 [3, 7, 11, 10, 12, 9, 11, 5, 13, 12, 4, 10, 4, 6, 15, 5, 14, 13, 14, 16, 15, 1, 2, 1, 17, 6, 16, 18, 2, 17, 18, 8, 8, 3, 7, 9]),
([11, 16, 5, 10, 12, 11, 3, 4, 14, 15, 15, 17, 1, 16, 2, 2, 3, 13, 12, 13, 4, 9, 10, 14, 8, 18, 17, 7, 7, 8, 6, 1, 5, 6, 9, 18],
 [4, 6, 5, 8, 11, 5, 9, 10, 10, 3, 9, 7, 13, 8, 4, 12, 12, 2, 11, 15, 3, 14, 13, 14, 1, 16, 1, 15, 17, 2, 16, 18, 6, 7, 17, 18]),
([3, 4, 13, 15, 18, 5, 4, 16, 17, 16, 17, 2, 6, 8, 3, 9, 5, 12, 14, 13, 14, 11, 15, 10, 10, 11, 1, 9, 12, 6, 1, 7, 7, 2, 8, 18],
 [1, 5, 11, 12, 13, 12, 14, 13, 6, 6, 4, 16, 5, 10, 11, 15, 15, 14, 2, 17, 16, 3, 7, 2, 9, 10, 1, 8, 8, 9, 3, 18, 4, 7, 17, 18]),
([3, 15, 16, 17, 16, 18, 5, 17, 7, 4, 18, 4, 14, 15, 12, 13, 13, 6, 2, 3, 14, 9, 1, 1, 11, 2, 12, 10, 6, 10, 11, 8, 7, 5, 8, 9],
 [15, 13, 14, 8, 1, 10, 9, 9, 2, 4, 14, 16, 15, 3, 3, 17, 11, 2, 12, 11, 12, 6, 10, 1, 13, 5, 5, 16, 18, 4, 7, 7, 17, 18, 6, 8]),
([3, 4, 14, 2, 17, 3, 15, 12, 18, 5, 16, 15, 16, 4, 11, 13, 12, 13, 14, 17, 1, 6, 8, 6, 9, 1, 10, 9, 10, 7, 2, 7, 8, 5, 11, 18],
 [3, 4, 6, 10, 11, 12, 11, 13, 12, 5, 5, 15, 4, 14, 7, 13, 9, 10, 14, 8, 8, 16, 15, 1, 6, 9, 2, 2, 17, 3, 16, 18, 1, 7, 17, 18]),
([6, 7, 8, 18, 7, 2, 9, 8, 1, 1, 4, 5, 14, 17, 5, 12, 16, 6, 15, 15, 10, 9, 3, 4, 13, 3, 13, 11, 2, 10, 11, 12, 14, 16, 17, 18],
 [1, 2, 3, 7, 8, 11, 2, 9, 8, 10, 9, 5, 10, 4, 13, 3, 12, 11, 12, 15, 4, 14, 13, 14, 6, 16, 6, 15, 17, 5, 16, 18, 1, 7, 17, 18])
]

def ryan15_seeds_for_n(n: int, take: int) -> List[Instance]:
    if n != 18:
        return []
    out=[]
    for H,V in _RYAN_18[:max(0, min(take, len(_RYAN_18)))]:
        h,v = canonicalize(H,V)
        out.append((h,v))
    return out

# ------------- PatternBoost loop -------------
def elites_for_n(elites: List[Tuple[float, Seq, Seq]], n: int) -> List[Tuple[float, Seq, Seq]]:
    return [e for e in elites if e[1] and max(e[1]) == n]

def recombine_seeds(elites: List[Tuple[float, Seq, Seq]], k: int,
                    rng: random.Random, n: int) -> List[Instance]:
    pool = elites_for_n(elites, n)
    if not pool:
        return []
    out = []
    for _ in range(k):
        _, H1, _ = rng.choice(pool)
        _, _, V2 = rng.choice(pool)
        h, v = canonicalize(H1, V2)
        out.append((h, v))
    return out

def ns_sequence(n_start: int, n_target: int, step: int) -> List[int]:
    out = []
    n = n_start
    while n <= n_target:
        out.append(n)
        n += step
    return out

def make_run_dirs(out_root: str, seed: int, n_list: List[int]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_root, f"misr_run-{ts}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for n in n_list:
        os.makedirs(os.path.join(run_dir, f"n{n:02d}"), exist_ok=True)
    return run_dir

def run_patternboost(
    seed: int = 123,
    n_start: int = 8,
    n_target: int = 32,
    rounds_per_n: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    elites_to_train: int = 96,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    grb_threads: int = 0,
    lift_step: int = 3,
    out_root: str = "runs",
    seed_pkl: str = "",
    seed_top_k: int = 4,
    # model knobs
    d_model: int = 384,
    n_head: int = 8,
    n_layer: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.1,
    aux_lambda: float = 0.25,
    aux_off: bool = False,
    # seed injections
    ryan15_seeds: bool = False,
    ryan15_take: int = 8,
    chuzhoy_seeds_flag: bool = False,
    chuzhoy_variants: int = 8,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    n_list = ns_sequence(n_start, n_target, lift_step)
    run_dir = make_run_dirs(out_root, seed, n_list)
    print(f"[run_dir] {run_dir}")

    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        json.dump({
            "seed": seed, "n_start": n_start, "n_target": n_target,
            "rounds_per_n": rounds_per_n, "seeds_per_round": seeds_per_round,
            "local_time_per_seed": local_time_per_seed,
            "elites_to_train": elites_to_train, "batch_size": batch_size,
            "train_steps_per_round": train_steps_per_round,
            "temperature": temperature, "top_p": top_p,
            "alpha_lp": alpha_lp, "beta_ilp": beta_ilp,
            "grb_threads": grb_threads, "lift_step": lift_step,
            "d_model": d_model, "n_head": n_head, "n_layer": n_layer,
            "ff_mult": ff_mult, "dropout": dropout, "aux_lambda": aux_lambda,
            "ryan15_seeds": ryan15_seeds, "ryan15_take": ryan15_take,
            "chuzhoy_seeds_flag": chuzhoy_seeds_flag, "chuzhoy_variants": chuzhoy_variants,
        }, f, indent=2)

    model = BetterGPT(d=d_model, nhead=n_head, nlayers=n_layer, ff_mult=ff_mult, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    elites: List[Tuple[float, Seq, Seq]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    n = n_start
    best_overall = 0.0

    # initial seeds
    seeds: List[Instance] = []
    # optional file seeds
    if seed_pkl:
        injected = load_seeds_from_pkl(seed_pkl, n, top_k=seed_top_k)
        if injected:
            print(f"[seed_inject/pkl] {len(injected)} items from {seed_pkl} for n={n}")
            seeds.extend(injected[:seed_top_k])

    # optional Chuzhoy seeds
    if chuzhoy_seeds_flag and seeds_for_n_from_chuzhoy is not None:
        cz = seeds_for_n_from_chuzhoy(n, variants=chuzhoy_variants)
        if cz:
            print(f"[seed_inject/chuzhoy] {len(cz)} for n={n}")
            for (H,V) in cz:
                seeds.append(canonicalize(H,V))

    # optional Ryan seeds
    if ryan15_seeds:
        rs = ryan15_seeds_for_n(n, ryan15_take)
        if rs:
            print(f"[seed_inject/ryan15] {len(rs)} for n={n}")
            seeds.extend(rs)

    # fill remainder
    base = motif_seeds(n)
    rng.shuffle(base)
    while len(seeds) < seeds_per_round and base:
        m = base.pop()
        seeds.append((m[:], random_valid_seq(n, rng)))
    while len(seeds) < seeds_per_round:
        seeds.append((random_valid_seq(n, rng), random_valid_seq(n, rng)))

    while n <= n_target:
        n_dir = os.path.join(run_dir, f"n{n:02d}")
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")
        for r in range(rounds_per_n):
            # 1) local search
            for (H,V) in seeds:
                es, best = local_search((H,V), time_budget_s=local_time_per_seed, rng=rng,
                                        alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads,
                                        elite_size=64, neighbor_k=96)
                for (score, h, v) in es:
                    push_elite(score, h, v)
                if best is not None:
                    best_overall = max(best_overall, best)
            print(f"[round {r+1}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f}")

            # 2) train transformer on elites
            topk = elites[:max(elites_to_train, min(32, len(elites)))]
            if topk:
                last_loss = None
                for _ in range(train_steps_per_round):
                    batch = make_batch(topk, min(batch_size, len(topk)), rng)
                    last_loss = train_one_step(model, opt, batch, aux_adj=(not aux_off), aux_lambda=aux_lambda)
                print(f"   trained {train_steps_per_round} steps, last CE ~ {last_loss:.3f}")

            # 3) new seeds
            new_seeds: List[Instance] = []
            new_seeds.extend(recombine_seeds(elites, k=max(1, seeds_per_round//4), rng=rng, n=n))
            while len(new_seeds) < seeds_per_round:
                elite_mut = (rng.random() < 0.25)
                pool_n = elites_for_n(elites, n)
                if elite_mut and pool_n:
                    _, h, v = rng.choice(pool_n[:min(64, len(pool_n))])
                    h = h[:]; v = v[:]
                    for S in (h, v):
                        if rng.random() < 0.6:
                            i, j = rng.randrange(len(S)), rng.randrange(len(S))
                            S[i], S[j] = S[j], S[i]
                    new_seeds.append((h, v))
                else:
                    h, v = sample_model(model, n, temperature=temperature, top_p=top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        h, v = random_valid_seq(n, rng), random_valid_seq(n, rng)
                    if rng.random() < 0.35:
                        for S in (h, v):
                            a = rng.randrange(len(S)); b = rng.randrange(len(S))
                            a, b = min(a,b), max(a,b)
                            if a != b:
                                S[a:b+1] = list(reversed(S[a:b+1]))
                    new_seeds.append((h, v))
            # small motif refresh
            mix = []
            RH, RV = motif_corner_combo(n)
            mix.append((RH[:], RV[:])); mix.append((RV[:], RH[:]))
            motifs = motif_seeds(n)[:2]
            for m in motifs:
                mix.append((m[:], motif_rainbow(n)))
            for i in range(min(len(mix), max(2, seeds_per_round//8))):
                new_seeds[i] = mix[i]
            seeds = new_seeds

            # 4) save per-round elites for n
            elites_n_only = elites_for_n(elites, n)
            ts_round = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_name = f"round{r+1:02d}_elites_n{n:02d}_{ts_round}.pkl"
            pkl_path = os.path.join(n_dir, pkl_name)
            with open(pkl_path, "wb") as f:
                pickle.dump(elites_n_only, f)
            with open(os.path.join(n_dir, "LATEST.txt"), "w") as f:
                f.write(pkl_name + "\n")

        # curriculum lift
        if elites and n < n_target:
            n_next = n + lift_step
            lifted=[]
            for _, h, v in elites[:min(96, len(elites))]:
                h2, v2 = lift_instance(h, v, n_next, rng)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
        else:
            break

    # summary
    print("\n=== BEST ELITES ===")
    for i,(score,h,v) in enumerate(elites[:10]):
        print(f"#{i+1} ratio={score:.4f}  n={max(h)}")
    final_path = os.path.join(run_dir, "final_elites.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(elites[:256], f)
    print(f"Saved top elites to {final_path}")
    return elites, run_dir

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--alpha_lp", type=float, default=0.25)
    ap.add_argument("--beta_ilp", type=float, default=0.20)
    ap.add_argument("--grb_threads", type=int, default=0)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--seed_pkl", type=str, default="")
    ap.add_argument("--seed_top_k", type=int, default=4)
    # model
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux_lambda", type=float, default=0.25)
    ap.add_argument("--aux_off", action="store_true")
    # injections
    ap.add_argument("--ryan15_seeds", action="store_true")
    ap.add_argument("--ryan15_take", type=int, default=8)
    ap.add_argument("--chuzhoy_seeds", action="store_true")
    ap.add_argument("--chuzhoy_variants", type=int, default=8)

    args = ap.parse_args()
    print(f"Device: {DEVICE}")

    _ = run_patternboost(
        seed=args.seed,
        n_start=args.n_start, n_target=args.n_target, rounds_per_n=args.rounds_per_n,
        seeds_per_round=args.seeds_per_round, local_time_per_seed=args.local_time_per_seed,
        elites_to_train=args.elites_to_train, batch_size=args.batch_size,
        train_steps_per_round=args.train_steps_per_round,
        temperature=args.temperature, top_p=args.top_p,
        alpha_lp=args.alpha_lp, beta_ilp=args.beta_ilp,
        grb_threads=args.grb_threads, lift_step=args.lift_step, out_root=args.out_root,
        seed_pkl=args.seed_pkl, seed_top_k=args.seed_top_k,
        d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
        ff_mult=args.ff_mult, dropout=args.dropout,
        aux_lambda=args.aux_lambda, aux_off=args.aux_off,
        ryan15_seeds=args.ryan15_seeds, ryan15_take=args.ryan15_take,
        chuzhoy_seeds_flag=args.chuzhoy_seeds, chuzhoy_variants=args.chuzhoy_variants,
    )

if __name__ == "__main__":
    main()
