import matplotlib.pyplot as plt
import numpy as np

def plot_coverage(layout,cov,sensors,out_png,show_blind_spots=True):
    obs=np.zeros((layout.height,layout.width))
    for r in getattr(layout,"obstacles",[]) or []:
        obs[int(r.y0):int(r.y1),int(r.x0):int(r.x1)]=1
    plt.figure(figsize=(8,5))
    plt.imshow(obs,cmap="gray",alpha=0.35)
    plt.imshow(cov,alpha=0.65)
    xs=[s.x for s in sensors]; ys=[s.y for s in sensors]
    if xs: plt.scatter(xs,ys,s=60)
    if show_blind_spots:
        by,bx=np.where((cov<=0)&(obs==0))
        if len(bx): plt.scatter(bx,by,s=6,marker="x")
    plt.tight_layout(); plt.savefig(out_png,dpi=200); plt.close()
