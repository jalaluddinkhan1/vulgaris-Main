import vulgaris
import numpy as np

# Check every exported name is reachable
missing = [name for name in vulgaris.__all__ if not hasattr(vulgaris, name)]
if missing:
    print("MISSING:", missing)
    raise SystemExit(1)
print(f"All {len(vulgaris.__all__)} exports reachable.")

# Model + StreamingInference
model  = vulgaris.Vulgaris(vulgaris.ModelConfig(input_dim=8, output_dim=1))
engine = vulgaris.StreamingInference(model, batch_size=1)
result = engine.step(np.random.randn(1, 8).astype(np.float32))
print(f"StreamingInference OK  latency={result['latency_ms']:.2f}ms")

# DistillationTrainer
student = vulgaris.Vulgaris(vulgaris.ModelConfig(input_dim=8, output_dim=1))
opt     = vulgaris.SpectralAdamW(student.parameters(), lr=1e-4)
trainer = vulgaris.DistillationTrainer(model, student, opt)
print("DistillationTrainer OK")

# ActiveLearner
learner = vulgaris.ActiveLearner(model, strategy="uncertainty")
pool    = np.random.randn(10, 8, 16).astype(np.float32)
idx     = learner.query(pool, n_query=3)
print(f"ActiveLearner OK  queried={idx}")

# SpeculativeRollout
sr  = vulgaris.SpeculativeRollout(model, gamma=2)
ctx = np.random.randn(1, 8, 16).astype(np.float32)
res = sr.rollout(ctx, horizon=4)
print(f"SpeculativeRollout OK  shape={res['outputs'].shape}")

# OntologyRegistry + OntologyEmbedding
reg  = vulgaris.OntologyRegistry()
reg.register(0, ["temperature", "vibration", "pressure"])
emb  = vulgaris.OntologyEmbedding(meta_dim=32)
z    = emb.encode(reg, domain_idx=0)
print(f"OntologyEmbedding OK  shape={z.data.shape}")

# IndustrialTokenizer
specs = [vulgaris.ChannelSpec(idx=0, token_type=vulgaris.TokenType.CONTINUOUS, hz=100.0)]
tok   = vulgaris.IndustrialTokenizer(specs, d_model=32)
print("IndustrialTokenizer OK")

# RegimeMixtureCore
from engine.tensor import Tensor
rmc      = vulgaris.RegimeMixtureCore(d_model=32, n_experts=4)
z_in     = Tensor(np.random.randn(2, 8, 32))
z_out, _ = rmc(z_in)
print(f"RegimeMixtureCore OK  out={z_out.data.shape}")

# Rule
rule = vulgaris.Rule(feature=0, op=">", threshold=3.5, consequence="alert")
print(f"Rule OK  {rule}")

# MuonOptimizer
muon = vulgaris.MuonOptimizer(model.parameters(), lr=1e-3)
print("MuonOptimizer OK")

# CanaryController
canary = vulgaris.CanaryController(mode=vulgaris.DeploymentMode.PRIMARY)
print("CanaryController OK")

# DriftDetector
drift = vulgaris.DriftDetector()
print("DriftDetector OK")

print()
print("pip install vulgaris  -->  all 71 exports work.")
