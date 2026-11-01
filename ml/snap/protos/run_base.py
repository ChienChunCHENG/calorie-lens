import json, time
import eval_all as ea, protolib as pl, fit
ea.setup(); T = fit.load_text(); t0 = time.time()
R = ea.run_pipeline(pl.TextOnly(T)); print({k: round(v, 3) if isinstance(v, float) else v for k, v in ea.summarize(R).items()}, f"{time.time()-t0:.0f}s")
ea.save_cache()
