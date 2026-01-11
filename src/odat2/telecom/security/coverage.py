import numpy as np
from dataclasses import dataclass
from math import atan2, degrees
from .models import SensorSpec

@dataclass(frozen=True)
class CoverageSummary:
    width: int
    height: int
    covered_cells: int
    uncovered_cells: int
    coverage_pct: float
    sensors: int
    notes: list
    single_covered_cells: int

def _angle_diff(a,b):
    d=(a-b)%360
    return d-360 if d>180 else d

def _bresenham(x0,y0,x1,y1):
    pts=[]
    dx,dy=abs(x1-x0),abs(y1-y0)
    sx=1 if x0<x1 else -1
    sy=1 if y0<y1 else -1
    err=dx-dy
    x,y=x0,y0
    while True:
        pts.append((x,y))
        if x==x1 and y==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x+=sx
        if e2<dx: err+=dx; y+=sy
    return pts

class CoverageAnalyzer:
    def __init__(self, layout, sensors):
        self.layout=layout
        self.sensors=sensors
        self.w=int(layout.width); self.h=int(layout.height)
        self.obs=np.zeros((self.h,self.w),dtype=np.uint8)
        for r in getattr(layout,"obstacles",[]) or []:
            self.obs[int(r.y0):int(r.y1),int(r.x0):int(r.x1)]=1

    def coverage_grid(self):
        cov=np.zeros((self.h,self.w),dtype=float)
        for s in self.sensors:
            sx,sy=int(s.x),int(s.y)
            for y in range(self.h):
                for x in range(self.w):
                    if self.obs[y,x]: continue
                    dx,dy=x-sx,y-sy
                    if dx*dx+dy*dy> s.range_cells*s.range_cells: continue
                    ang=(degrees(atan2(dy,dx))+360)%360
                    if s.fov_deg<360 and abs(_angle_diff(ang,s.heading_deg))>s.fov_deg/2: continue
                    blocked=False
                    for px,py in _bresenham(sx,sy,x,y)[1:]:
                        if self.obs[py,px]: blocked=True; break
                    if not blocked: cov[y,x]+=1
        return cov

    def blind_spots(self,cov):
        return [(x,y) for y in range(self.h) for x in range(self.w) if cov[y,x]<=0 and not self.obs[y,x]]

    def summary(self,cov):
        free=int((self.obs==0).sum())
        covered=int(((cov>0)&(self.obs==0)).sum())
        single=int(((cov==1)&(self.obs==0)).sum())
        return CoverageSummary(self.w,self.h,covered,free-covered,0 if free==0 else 100*covered/free,len(self.sensors),[],single)
