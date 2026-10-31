"""Frozen tiny F32 GGUF fixture for selector correctness, never a quality benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

VOCAB=16384
HIDDEN=32
LAYERS=2
HEADS=4
KV_HEADS=2
FFN=64
CONTEXT=128

def string(value):
    data=value if isinstance(value,bytes) else value.encode('utf-8')
    return struct.pack('<Q',len(data))+data

def scalar(value,kind):
    return struct.pack('<I',kind)+({4:lambda:struct.pack('<I',value),6:lambda:struct.pack('<f',value),8:lambda:string(value)}[kind]())

def build(output):
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('fixture artifacts are immutable')
    words=['<unk>','<|endoftext|>','<|user|>','<|assistant|>']
    words += [bytes([i]) for i in range(256)]
    words += [b'\0\0']
    words += ['token'+str(i) for i in range(len(words),VOCAB)]
    meta={'general.architecture':('llama',8),'general.name':('e10 synthetic selector fixture, not registered pick',8),'general.alignment':(32,4),
          'llama.vocab_size':(VOCAB,4),'llama.context_length':(CONTEXT,4),'llama.embedding_length':(HIDDEN,4),'llama.block_count':(LAYERS,4),
          'llama.feed_forward_length':(FFN,4),'llama.attention.head_count':(HEADS,4),'llama.attention.head_count_kv':(KV_HEADS,4),
          'llama.attention.layer_norm_rms_epsilon':(1e-5,6),'llama.rope.dimension_count':(HIDDEN//HEADS,4),
          'llama.rope.freq_base':(10000.,6),'tokenizer.ggml.model':('bpe',8),'tokenizer.ggml.bos_token_id':(2,4),'tokenizer.ggml.eos_token_id':(1,4)}
    names=[('token_embd.weight',(HIDDEN,VOCAB),False),('output_norm.weight',(HIDDEN,),True),('output.weight',(HIDDEN,VOCAB),False)]
    for layer in range(LAYERS):
        root='blk.'+str(layer)+'.'
        names += [(root+'attn_norm.weight',(HIDDEN,),True),(root+'ffn_norm.weight',(HIDDEN,),True)]
        names += [(root+n+'.weight',dims,False) for n,dims in [('attn_q',(HIDDEN,HIDDEN)),('attn_k',(HIDDEN,KV_HEADS*(HIDDEN//HEADS))),('attn_v',(HIDDEN,KV_HEADS*(HIDDEN//HEADS))),('attn_output',(HIDDEN,HIDDEN)),('ffn_gate',(HIDDEN,FFN)),('ffn_up',(HIDDEN,FFN)),('ffn_down',(FFN,HIDDEN))]]
    metadata=b''.join(string(k)+scalar(v,t) for k,(v,t) in meta.items())
    metadata += string('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,VOCAB)+b''.join(map(string,words))
    metadata += string('tokenizer.ggml.merges')+struct.pack('<IIQ',9,8,1)+string('4 4')
    infos=[];data=bytearray();seed=20261031
    for name,dims,norm in names:
        while len(data)%32:data.append(0)
        offset=len(data);count=1
        for d in dims:count*=d
        values=[]
        for _ in range(count):
            seed=(1664525*seed+1013904223)&0xffffffff
            values.append(1. if norm else (((seed>>8)&65535)-32768)/262144.)
        data.extend(struct.pack('<'+str(count)+'f',*values))
        infos.append(string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',0,offset))
    prefix=b'GGUF'+struct.pack('<IQQ',3,len(names),len(meta)+2)+metadata+b''.join(infos)
    prefix+=b'\0'*((-len(prefix))%32)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial');temporary.write_bytes(prefix+data);temporary.replace(output)
    receipt={'scope':'Synthetic F32 llama only; registered pick_e1_L16 and adoption gates remain pending','sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'geometry':{'vocab':VOCAB,'hidden':HIDDEN,'layers':LAYERS,'heads':HEADS,'kv_heads':KV_HEADS,'ffn':FFN,'context':CONTEXT},'tensor_inventory':[{'name':n,'gguf_dimensions':list(d)} for n,d,_ in names],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tensor_generation':'uint32 LCG1664525/1013904223, seed20261031; fixed-point exactly representable F32; RMS scales1'}
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(build(a.output)))
