import numpy as np, time, json
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
a = EmbodiedHuman(SimConfig(duration=2.0, log_every=10), vision=True)
for _ in range(1500): a.step()
f = a.frame
out = {"complexity": a.complexity.name if hasattr(a,'complexity') else None}
out["tactile"] = list(f.tactile.shape); out["proprio"]=list(f.proprio.shape)
for k in ("vestibular","visual","auditory","olfactory","gustatory","chemo_summary","skin_temperature"):
    out[k]=list(getattr(f,k).shape)
out["ext"]={k:list(v.shape) for k,v in f.ext.items()}
out["intero"]=list(a.interoception.s.shape)
af=a.affect_frame; out["emotions"]=list(af.emotions.shape); out["neuromod"]=list(af.neuromodulators.shape); out["appraisal"]=list(af.appraisal.shape)
out["drives"]=list(a.drive_frame.level.shape)
out["n_actuators"]=a.meta.n_actuators
out["n_patches"]=len(__import__('embodied_human.skin',fromlist=['x']).PATCH_NAMES)
out["latent_blocks"]=a.latent_spec.describe() if hasattr(a.latent_spec,'describe') else None
out["inner_counts"]=a.inner.counts()
out["attrs"]=[k for k in dir(a) if not k.startswith('_')][:200]
print(json.dumps(out,default=str,indent=1))
