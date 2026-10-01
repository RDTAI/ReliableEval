"""Small, dependency-light reliability evaluation primitives."""
from __future__ import annotations
import numpy as np

def _check(p,y):
 p=np.asarray(p,float); y=np.asarray(y,int)
 if p.ndim!=2 or y.ndim!=1 or len(p)!=len(y): raise ValueError('probabilities and labels have incompatible shapes')
 if np.any(p<0) or np.any(~np.isfinite(p)): raise ValueError('probabilities must be finite and non-negative')
 p=p/p.sum(1,keepdims=True); return p,y

def classification_metrics(probabilities,labels,n_bins=10):
 p,y=_check(probabilities,labels); pred=p.argmax(1); conf=p.max(1); bins=np.linspace(0,1,n_bins+1); ece=0.
 for lo,hi in zip(bins[:-1],bins[1:]):
  m=(conf>lo)&(conf<=hi if hi<1 else conf<=hi)
  if m.any(): ece += m.mean()*abs((pred[m]==y[m]).mean()-conf[m].mean())
 one=np.eye(p.shape[1])[y]
 return {'n':int(len(y)),'accuracy':float((pred==y).mean()),'mean_confidence':float(conf.mean()),'ece':float(ece),'brier':float(((p-one)**2).sum(1).mean()),'class_accuracy':{str(c):float((pred[y==c]==c).mean()) if (y==c).any() else None for c in range(p.shape[1])}}

def conformal_threshold(calibration_probabilities,calibration_labels,alpha=.1):
 p,y=_check(calibration_probabilities,calibration_labels); scores=1-p[np.arange(len(y)),y]; rank=int(np.ceil((len(y)+1)*(1-alpha))); return float(np.inf if rank>len(scores) else np.sort(scores)[rank-1])

def conformal_sets(probabilities,threshold):
 p=np.asarray(probabilities,float); return p >= (1-float(threshold))

def evaluate_conformal(calibration_probabilities,calibration_labels,test_probabilities,test_labels,alpha=.1):
 p,y=_check(test_probabilities,test_labels); q=conformal_threshold(calibration_probabilities,calibration_labels,alpha); sets=conformal_sets(p,q); return {'alpha':float(alpha),'target_coverage':float(1-alpha),'empirical_coverage':float(sets[np.arange(len(y)),y].mean()),'mean_set_size':float(sets.sum(1).mean()),'singleton_rate':float((sets.sum(1)==1).mean()),'empty_rate':float((sets.sum(1)==0).mean()),'threshold':q}

def energy_score(logits,temperature=1.): z=np.asarray(logits,float)/temperature; return -temperature*np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))-z.max(1)

def risk_coverage_curve(probabilities,labels,steps=20):
 p,y=_check(probabilities,labels); conf=p.max(1); pred=p.argmax(1); order=np.argsort(-conf); out=[]
 for coverage in np.linspace(1/steps,1,steps):
  n=max(1,int(np.ceil(coverage*len(y)))); out.append({'coverage':float(n/len(y)),'risk':float(1-(pred[order[:n]]==y[order[:n]]).mean())})
 return out

def build_report(test_probabilities,test_labels,calibration_probabilities=None,calibration_labels=None,alpha=.1):
 r={'classification':classification_metrics(test_probabilities,test_labels),'risk_coverage':risk_coverage_curve(test_probabilities,test_labels)}
 if calibration_probabilities is not None and calibration_labels is not None:r['conformal']=evaluate_conformal(calibration_probabilities,calibration_labels,test_probabilities,test_labels,alpha)
 return r
