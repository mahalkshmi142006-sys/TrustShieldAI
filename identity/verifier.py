import cv2,numpy as np
def desc(p):
    im=cv2.imread(str(p))
    if im is None:raise ValueError("Could not read image.")
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml"); fs=cas.detectMultiScale(g,1.1,5)
    if len(fs)==0:return None,0
    x,y,w,h=max(fs,key=lambda f:f[2]*f[3]); c=cv2.resize(g[y:y+h,x:x+w],(128,128)); hist=cv2.normalize(cv2.calcHist([c],[0],None,[64],[0,256]),None).flatten()
    return hist,len(fs)
def verify_faces(a,b):
    x,na=desc(a); y,nb=desc(b)
    if x is None or y is None:return {"status":"UNCERTAIN","similarity":0,"message":"A clear face was not detected in both images."}
    score=round(max(0,min(100,float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))*100)),2)
    return {"status":"MATCH" if score>=90 else ("REVIEW" if score>=75 else "NO MATCH"),"similarity":score,"reference_faces":na,"probe_faces":nb,"method":"Histogram face descriptor prototype; not biometric-grade identification"}
