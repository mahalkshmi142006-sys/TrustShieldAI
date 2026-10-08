"""Baseline evaluation of the current (rule + TF-IDF) hallucination engine."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hallucination.engine import analyze_answer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
cases = json.load(open(ROOT/"data"/"hallucination_eval.json"))
yt, yp, wrong = [], [], []
for c in cases:
    r = analyze_answer(c["question"], c["response"], ROOT/"data"/"evidence.json")
    p = "CONTRADICTED" if r["contradicted"]>0 else ("SUPPORTED" if r["supported"]==(r["supported"]+r["uncertain"]+r["contradicted"]) else "UNCERTAIN")
    yt.append(c["expected"]); yp.append(p)
    if p!=c["expected"]: wrong.append((c["expected"],p,c["response"]))
L=["SUPPORTED","CONTRADICTED","UNCERTAIN"]
pr,rc,f1,_=precision_recall_fscore_support(yt,yp,labels=L,zero_division=0)
res={"n":len(cases),"accuracy":round(accuracy_score(yt,yp),4),"macro_f1":round(float(f1.mean()),4),
     "per_class_recall":dict(zip(L,[round(float(x),3) for x in rc])),"confusion_matrix(rows=true,cols=pred)":confusion_matrix(yt,yp,labels=L).tolist()}
print(json.dumps(res,indent=1)); print("\nMisclassified:")
for w in wrong: print(w)
json.dump(res,open(ROOT/"experiments"/"hallucination_baseline_results.json","w"),indent=2)
