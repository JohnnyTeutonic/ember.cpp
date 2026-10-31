import subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def body(s,n):
 a=s.index(n);b=s.index('{',a);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
class SourceTests(unittest.TestCase):
 def test_no_unmeasured_atomic_production_paths(self):
  s=(ROOT/'cuda_kernels.cu').read_text()
  for name in ('rmsnorm_optimized_kernel','softmax_optimized_kernel','attention_optimized_kernel'):
   self.assertNotIn('__global__ void '+name+'(',s);self.assertNotIn(name+'<<<',s)
 def test_all_wrapper_routes_ordered_without_flag(self):
  s=(ROOT/'cuda_kernels.cu').read_text()
  for name,target in [('void rmsnorm_vector_cuda_optimized(','rmsnorm_optimized_det_kernel<<<1, 256'),('void softmax_vector_cuda_optimized(','softmax_optimized_det_kernel<<<1, 256'),('void attention_cuda_optimized(','attention_optimized_det_kernel<<<'),('void attention_devpos_cuda(','attention_optimized_det_kernel<<<')]:
   self.assertIn(name,s)
   wrapper=body(s,name);self.assertIn(target,wrapper);self.assertNotIn('ember::deterministic()',wrapper)
  self.assertEqual(s.count('attention_optimized_det_kernel<<<'),2)
 def test_ordered_existing_arithmetic_unchanged(self):
  old=subprocess.check_output(['git','-C',str(ROOT),'show','1b2789e:cuda_kernels.cu'],text=True);new=(ROOT/'cuda_kernels.cu').read_text()
  for name in ('rmsnorm_optimized_det_kernel','attention_optimized_det_kernel'):
   a=body(old,'__global__ void '+name+'(');b=body(new,'__global__ void '+name+'(').replace('\n    ember::require_single_block_reduction_launch();','')
   self.assertEqual(a,b)
 def test_guard_called_at_kernel_entry(self):
  s=(ROOT/'cuda_kernels.cu').read_text()
  for name in ('rmsnorm_optimized_det_kernel','softmax_optimized_det_kernel'):
   k=body(s,'__global__ void '+name+'(');self.assertTrue(k[k.index('{')+1:].lstrip().startswith('ember::require_single_block_reduction_launch();'))
  h=(ROOT/'reduction_launch.h').read_text();self.assertIn('__trap();',h);self.assertNotIn('assert(',h.replace('unlike assert(),',''))
 def test_release_active_geometry_predicate(self):
  source='#include "reduction_launch.h"\nint main(){using ember::valid_single_block_reduction_launch; if(!valid_single_block_reduction_launch(1,1,1,256,1,1))return 1; unsigned bad[][6]={{2,1,1,256,1,1},{1,2,1,256,1,1},{1,1,2,256,1,1},{1,1,1,31,1,1},{1,1,1,256,2,1},{1,1,1,256,1,2},{1,1,1,1056,1,1}};for(auto &x:bad)if(valid_single_block_reduction_launch(x[0],x[1],x[2],x[3],x[4],x[5]))return 2;return 0;}'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'test.cpp').write_text(source);subprocess.run(['g++','-std=c++17','-DNDEBUG','-O2','-I'+str(ROOT),str(p/'test.cpp'),'-o',str(p/'test')],check=True);subprocess.run([str(p/'test')],check=True)
if __name__=='__main__':unittest.main()
