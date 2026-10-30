import sys,pathlib,hashlib,json,importlib
sys.path.insert(0,sys.argv[1])
mod=importlib.import_module('tinyllama_cpp.tinyllama_bindings')
assert mod.cuda_compiled is False
for api in ('ModelConfig','TinyLlamaSession'): assert callable(getattr(mod,api))
for method in ('generate','generate_batch'): assert callable(getattr(mod.TinyLlamaSession,method))
cfg=mod.ModelConfig(); cfg.hidden_size=32; cfg.vocab_size=64
assert cfg.hidden_size==32 and cfg.vocab_size==64
print(json.dumps({'native_module_sha256':hashlib.sha256(pathlib.Path(mod.__file__).read_bytes()).hexdigest(),'cuda_compiled':mod.cuda_compiled,'configuration_roundtrip':True,'required_apis_callable':True}))
