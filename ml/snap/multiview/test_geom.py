import numpy as np, cv2, sys
sys.path.insert(0,'.')
import geom
K=geom.K_of(640,360)
for tilt,az,dist in [(40,0,45),(60,20,50),(25,-30,40),(70,0,60)]:
    # camera above table: plate normal n (to camera) in camera coords; build by rotating
    t=np.radians(tilt); n=np.array([np.sin(np.radians(az))*np.sin(t),-np.sin(t)*np.cos(np.radians(az))*0-np.sin(t), -np.cos(t)])
    n=np.array([0.0,-np.sin(t),-np.cos(t)])  # camera looks down by tilt from the plate normal; plate up = toward image top
    a,b=geom.plane_basis(n); c=np.array([3.0,2.0,dist])
    th=np.linspace(0,2*np.pi,200); P=c+13*(np.outer(np.cos(th),a)+np.outer(np.sin(th),b))
    uv=(K@P.T).T; uv=uv[:,:2]/uv[:,2:]
    (cx,cy),(w1,h1),ang=cv2.fitEllipse(uv.astype(np.float32))
    e=dict(cx=cx,cy=cy,major=max(w1,h1),minor=min(w1,h1),angle=ang if w1>=h1 else ang+90)
    r=geom.best_pose(e,K,26)
    print(tilt,dist,'->',None if r is None else (np.round(r[0],3),np.round(r[1],1),round(r[2],1)),'true n',np.round(n,3))
