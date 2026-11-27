"""exp05: sobreajuste y regularizacion sobre el modelo del starter.
Uso: python exp05_run.py cfg1 cfg2 ...   (EPOCHS y THREADS por variable de entorno)
Cada configuracion escribe results/exp05_parts/<cfg>.csv con una fila por epoca.
"""
import os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS', '1')))
EPOCHS = int(os.environ.get('EPOCHS', '6'))
STRIDE = int(os.environ.get('STRIDE', '1'))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, 'results', 'exp05_parts'); os.makedirs(OUT, exist_ok=True)

tr = np.load(os.path.join(DATA, 'train.npz')); st = np.load(os.path.join(DATA, 'static.npz'))
Xtr, Ytr, ttr = tr['X'], tr['Y'], tr['times'].astype(str)
dem = st['dem']; VARS = ['RH', 'T', 'U', 'V']
REF = np.array([25.360, 2.0652, 2.6776, 2.0876])  # RMSE_clim oficiales
hour = lambda t: np.array([int(k[-2:]) for k in t])

def climatology(Y, t):
    h = hour(t); return np.stack([Y[h == k].mean(0) for k in range(24)])

def windows(X, t):
    ts = pd.to_datetime(t, format='%Y%m%d%H'); pos = {k: i for i, k in enumerate(ts)}
    out = np.empty((len(X), 3) + X.shape[1:], np.float32)
    for i, k in enumerate(ts):
        for j, lag in enumerate([2, 1, 0]):
            idx = pos.get(k - pd.Timedelta(hours=lag), None)
            out[i, j] = X[idx] if idx is not None else X[i]
    return out

Wtr = windows(Xtr, ttr)
glat, glon = st['gfs_lats'], st['gfs_lons']; wlat, wlon = st['wrf_lats'], st['wrf_lons']
LAT, LON = np.meshgrid(wlat, wlon, indexing='ij')
gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
GRID = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)
DEM = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32)[None, None]

class Net(nn.Module):
    """Igual al starter; drop = Dropout2d tras cada bloque (0 = starter)."""
    def __init__(self, w=64, drop=0.0):
        super().__init__()
        D = (lambda: nn.Dropout2d(drop)) if drop > 0 else (lambda: nn.Identity())
        self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU(), D())
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(), D(),
                                 nn.Conv2d(w, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 4, 1))
    def forward(self, x, h):
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        ang = 2 * np.pi * h / 24
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
        f = self.enc(torch.cat([x, hh], 1))
        f = F.grid_sample(f, GRID.expand(B, -1, -1, -1), align_corners=True)
        f = torch.cat([f, DEM.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
        return self.dec(f).permute(0, 2, 3, 1)

BASE = dict(w=64, drop=0.0, wd=1e-4, noise=0.0, seed=0, cut='2024061600', val_end='9999')
CFGS = {
    'base':      {},
    'wd1e-2':    dict(wd=1e-2),
    'dropout01': dict(drop=0.1),
    'ruido005':  dict(noise=0.05),
    'ancho32':   dict(w=32),
    'semilla1':  dict(seed=1),
    'corte0601': dict(cut='2024060100', val_end='2024061600'),
}

def run(name):
    c = {**BASE, **CFGS[name]}
    torch.manual_seed(c['seed']); rng = np.random.default_rng(c['seed'])
    trn = ttr < c['cut']; val = (ttr >= c['cut']) & (ttr < c['val_end'])
    mu_x = Xtr[trn].mean((0, 1, 2)); sd_x = Xtr[trn].std((0, 1, 2))
    clim = climatology(Ytr[trn], ttr[trn]); asd = (Ytr[trn] - clim[hour(ttr[trn])]).std((0, 1, 2))
    clim_rmse = np.sqrt(((clim[hour(ttr[val])] - Ytr[val]) ** 2).mean((0, 1, 2)))
    CL = torch.tensor(clim, dtype=torch.float32); AS = torch.tensor(asd, dtype=torch.float32)
    prep = lambda W, t: (torch.tensor((W - mu_x) / sd_x, dtype=torch.float32), torch.tensor(hour(t)))
    def predict(m, idx):
        m.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s+128]; x, h = prep(Wtr[b], ttr[b])
                out.append((CL[h] + m(x, h.float()) * AS).numpy())
        return np.concatenate(out)
    def scores(pred, idx):
        rmse = np.sqrt(((pred - Ytr[idx]) ** 2).mean((0, 1, 2)))
        return 1 - (rmse / clim_rmse).mean(), 1 - (rmse / REF).mean(), rmse
    model = Net(c['w'], c['drop'])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=c['wd'])
    itr = np.where(trn)[0][::STRIDE]; ival = np.where(val)[0]  # STRIDE>1: submuestreo de train por tiempo de CPU
    itr_eval = np.sort(rng.choice(itr, size=min(len(ival), len(itr)), replace=False))  # muestra de train para la curva
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * int(np.ceil(len(itr) / 32)))
    rows = []
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(itr); t0 = time.time(); tl = []
        for s in range(0, len(itr), 32):
            b = itr[s:s+32]; x, h = prep(Wtr[b], ttr[b])
            if c['noise'] > 0: x = x + c['noise'] * torch.randn_like(x)
            y = (torch.tensor(Ytr[b], dtype=torch.float32) - CL[h]) / AS
            loss = F.mse_loss(model(x, h.float()), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        s_tr, _, _ = scores(predict(model, itr_eval), itr_eval)
        s_va, s_ref, rm = scores(predict(model, ival), ival)
        row = dict(config=name, epoca=ep + 1, loss_train=np.mean(tl), score_train=s_tr, score_val=s_va,
                   score_val_ref_oficial=s_ref, **{f'RMSE_{v}': rm[k] for k, v in enumerate(VARS)},
                   n_train=len(itr), stride=STRIDE, n_val=int(val.sum()), seg=time.time() - t0, **{k: str(v) for k, v in c.items()})
        rows.append(row); pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
        print(name, ep + 1, f'train {s_tr:.4f} val {s_va:.4f} ref {s_ref:.4f}', rm.round(3), f'{row["seg"]:.0f}s', flush=True)

if __name__ == '__main__':
    for n in sys.argv[1:]: run(n)
