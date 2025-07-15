import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

# -------------------------
# 1. 2D Sinusoidal Positional Encoding (righe + colonne)
# -------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model deve essere divisibile per 2"
        d_half = d_model // 2
        row_pos = torch.arange(height, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        pe_row = torch.zeros(height, d_half)
        pe_row[:, 0::2] = torch.sin(row_pos * div_term)
        pe_row[:, 1::2] = torch.cos(row_pos * div_term)
        col_pos = torch.arange(width, dtype=torch.float).unsqueeze(1)
        pe_col = torch.zeros(width, d_half)
        pe_col[:, 0::2] = torch.sin(col_pos * div_term)
        pe_col[:, 1::2] = torch.cos(col_pos * div_term)
        pe_row = pe_row.unsqueeze(1).expand(height, width, d_half)
        pe_col = pe_col.unsqueeze(0).expand(height, width, d_half)
        pe = torch.cat([pe_row, pe_col], dim=-1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.unsqueeze(0)

# -------------------------
# 2. Relative Positional Bias & 2D Attention
# -------------------------
class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, height: int, width: int):
        super().__init__()
        self.num_heads = num_heads
        self.height = height
        self.width = width
        size = (2*height-1)*(2*width-1)
        self.table = nn.Parameter(torch.zeros(size, num_heads))
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2,H,W)
        coords_flat = coords.view(2, -1)  # (2,N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2,N,N)
        rel[0] += height - 1
        rel[1] += width - 1
        rel[0] *= (2*width - 1)
        index = (rel[0] + rel[1]).view(-1)
        self.register_buffer('index', index)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self) -> torch.Tensor:
        N = self.height * self.width
        bias = self.table[self.index].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1)

class Attention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, height: int, width: int):
        super().__init__()
        assert dim % num_heads == 0, "dim deve essere divisibile per num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = RelativePositionBias2D(num_heads, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias()  # (heads, N, N)
        attn = (attn + bias.unsqueeze(0)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class EncoderBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, height: int, width: int,
                 dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention2D(dim, num_heads, height, width)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# -------------------------
# 3. Dataset con dimensione variabile input/output
# -------------------------
class MatrixDataset(Dataset):
    def __init__(self, challenges_path: str, solutions_path: str = None, train: bool = True,
                 H_max: int = 0, W_max: int = 0, patch_size: int = 4, H_out: int = 0, W_out: int = 0):
        print(f"[Dataset] Loading da {challenges_path}, train={train}")
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        self.train = train
        self.samples = []
        solutions = {}
        if not train and solutions_path:
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
        # Carica input-output coppie
        for cid, data in challenges.items():
            entries = data.get('train' if train else 'test', [])
            for idx, ex in enumerate(entries):
                inp = torch.tensor(ex['input'], dtype=torch.float32).unsqueeze(0)
                if not train and cid in solutions and idx < len(solutions[cid]):
                    out = torch.tensor(solutions[cid][idx], dtype=torch.float32).unsqueeze(0)
                else:
                    out = torch.tensor(ex.get('output', ex['input']), dtype=torch.float32).unsqueeze(0)
                self.samples.append((inp, out))
        self.H_max_in, self.W_max_in = H_max, W_max
        self.H_max_out, self.W_max_out = H_out, W_out
        self.patch_size = patch_size
        print(f"[Dataset] {len(self.samples)} campioni, pad IN=({H_max}×{W_max}), OUT=({H_out}×{W_out}), patch {patch_size}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, out = self.samples[idx]
        # pad input
        pad_w = self.W_max_in - inp.shape[2]
        pad_h = self.H_max_in - inp.shape[1]
        inp = F.pad(inp, (0, pad_w, 0, pad_h)).contiguous()
        # pad output
        pad_w2 = self.W_max_out - out.shape[2]
        pad_h2 = self.H_max_out - out.shape[1]
        out = F.pad(out, (0, pad_w2, 0, pad_h2)).contiguous()
        return inp, out

# -------------------------
# 4. Modello ViTRegressor con resume
# -------------------------
class ViTRegressor(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, dim_ff, dropout,
                 H_in, W_in, H_out, W_out, patch_size=4):
        super().__init__()
        # patch dims
        self.patch = patch_size
        self.H_p = (H_in - patch_size) // patch_size + 1
        self.W_p = (W_in - patch_size) // patch_size + 1
        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        self.encoder = nn.ModuleList([
            EncoderBlock2D(embed_dim, nhead, self.H_p, self.W_p, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.pos_enc = PositionalEncoding2D(embed_dim, H_out, W_out)
        dec_layer = nn.TransformerDecoderLayer(embed_dim, nhead, dim_ff, dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, inp: torch.Tensor, out_shape: tuple) -> torch.Tensor:
        B = inp.size(0)
        x = self.patch_embed(inp)
        x = x.flatten(2).transpose(1, 2)  # (B, N_in, d)
        for blk in self.encoder:
            x = blk(x)
        mem = x.transpose(0, 1)  # (N_in, B, d)
        H_out, W_out = out_shape
        tgt = torch.zeros(H_out*W_out, B, mem.size(2), device=inp.device)
        # positional
        po = self.pos_enc(tgt.transpose(0,1).view(B, H_out, W_out, -1))
        tgt = po.view(B, H_out*W_out, -1).transpose(0,1)
        out = self.decoder(tgt, mem)
        out = self.head(out).transpose(0,1).view(B, 1, H_out, W_out)
        return out

# -------------------------
# Utils: checkpoint, train, eval, infer
# -------------------------
def save_checkpoint(model, optimizer, epoch, loss, dir):
    os.makedirs(dir, exist_ok=True)
    f = os.path.join(dir, f'checkpoint_epoch_{epoch:02d}.pth')
    torch.save({'epoch':epoch,'model_state':model.state_dict(),
                'opt_state':optimizer.state_dict(),'loss':loss}, f)
    print(f"[Checkpoint] Epoch {epoch} salvato in {f}")

def train_epoch(m, loader, crit, opt, dev, e):
    m.train(); tot=0
    Hout, Wout = loader.dataset.H_max_out, loader.dataset.W_max_out
    for i,(inp,out) in enumerate(loader,1):
        inp,out=inp.to(dev),out.to(dev)
        opt.zero_grad(); pred=m(inp,(Hout,Wout))
        l=crit(pred,out);l.backward();opt.step()
        if i % 500 == 0:
            print(f"[Train] Epoch {e} Batch {i} Loss: {l.item():.6f}")
        tot+=l.item()
    return tot/len(loader)

def evaluate(m, loader, crit, dev, e):
    m.eval(); tot=0
    Hout, Wout = loader.dataset.H_max_out, loader.dataset.W_max_out
    with torch.no_grad():
        for inp,out in loader:
            inp,out=inp.to(dev),out.to(dev)
            tot+=crit(m(inp,(Hout,Wout)),out).item()
    #print(f"[Eval] Epoch {e} Loss: {tot/len(loader):.4f}")
    return tot/len(loader)

# -------------------------
# Main con resume
# -------------------------
def main():
    cpdir=f"./ckpt_{os.path.basename(__file__).split('.')[0]}"
    os.makedirs(cpdir,exist_ok=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Device: {dev}")
    # paths
    ch='./dataset/arc-agi_training_challenges_augmented.json'
    so='./dataset/arc-agi_training_solutions.json'
    # scan dimensions
    with open(ch) as f:chj=json.load(f)
    H_in=W_in=H_out=W_out=0
    for v in chj.values():
        for ex in v['train']+v['test']:
            H_in,W_in=max(H_in,len(ex['input'])),max(W_in,len(ex['input'][0]))
    sojson={}
    with open(so) as f:sojson=json.load(f)
    for k,v in sojson.items():
        for arr in v:
            H_out,W_out=max(H_out,len(arr)),max(W_out,len(arr[0]))
    # dataset
    train_ds=MatrixDataset(ch,so,True,H_in,W_in,4,H_out,W_out)
    val_ds=MatrixDataset(ch,so,False,H_in,W_in,4,H_out,W_out)
    ld_tr=DataLoader(train_ds,batch_size=1,shuffle=True)
    ld_va=DataLoader(val_ds,batch_size=1)
    # model
    m=ViTRegressor(64,8,6,256,0.1,H_in,W_in,H_out,W_out).to(dev)
    opt=torch.optim.Adam(m.parameters(),1e-4)
    crit=nn.MSELoss()
    # resume
    files=sorted([x for x in os.listdir(cpdir) if x.endswith('.pth')],
                 key=lambda x:int(x.split('_')[-1].split('.')[0]))
    start=1
    if files:
        cp=os.path.join(cpdir,files[-1]);print(f"[Resume] load {cp}")
        d=torch.load(cp);m.load_state_dict(d['model_state']);opt.load_state_dict(d['opt_state']);start=d['epoch']+1
        # loop


    for e in range(start, 501):
        train_loss = train_epoch(m, ld_tr, crit, opt, dev, e)
        val_loss = evaluate(m, ld_va, crit, dev, e)
        print(f"[Epoch {e}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        save_checkpoint(m, opt, e, val_loss, cpdir)

if __name__=='__main__': main()
