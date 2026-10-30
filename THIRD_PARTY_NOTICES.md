# Third-party notices

Ember's original code is MIT licensed; see LICENSE. Components and dependencies
below retain their own copyright and license terms.

## llama.cpp / ggml

Upstream: https://github.com/ggml-org/llama.cpp
Pinned tokenizer/table reference: b6000,
commit 4762ad7316dcdec20016ab5985fb46a27902204d.
License: https://github.com/ggml-org/llama.cpp/blob/4762ad7316dcdec20016ab5985fb46a27902204d/LICENSE

llama3_tokenizer.h identifies its port of unicode.cpp and llama-vocab.cpp;
llama3_unicode_tables.h identifies generated tables from unicode-data.cpp.
These arrived through paperkiln's corresponding headers. quantization.cpp also
identifies llama.cpp's scalar dequantization loop/layout references. The upstream
notice is retained here for those contributions. This does not claim that every
Ember CPU or CUDA kernel was copied from llama.cpp.

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## paperkiln

Upstream: https://github.com/JohnnyTeutonic/paperkiln
bpe_tokenizer.h, chat_template.h, llama3_tokenizer.h and
llama3_unicode_tables.h identify their paperkiln source pins in their headers.
The original paperkiln material is MIT licensed, copyright (c) 2024-2026 Jonathan
Reich; the complete matching notice and permission text is in LICENSE. The
llama.cpp-derived portions retain the additional upstream notice above.

## nlohmann/json

CMake uses a system installation or fetches v3.11.3 from
https://github.com/nlohmann/json. The following is its LICENSE.MIT from the
v3.11.3 source used by this build. System installations may have another version;
retain the notices supplied with the actual installed dependency as well.

MIT License 

Copyright (c) 2013-2022 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## cpp-httplib

CMake fetches v0.15.3 from https://github.com/yhirose/cpp-httplib.
The complete LICENSE from that source is retained below.

The MIT License (MIT)

Copyright (c) 2017 yhirose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## doxygen-awesome-css

The vendored documentation theme retains its full MIT notice in
[docs/doxygen/LICENSE.doxygen-awesome-css](docs/doxygen/LICENSE.doxygen-awesome-css)
(copyright 2021-2023 jothepro). That file remains part of the distribution.

## External runtimes and model assets

CUDA/cuBLAS, optional OpenSSL, Python dependencies and downloaded model assets
retain their respective terms. This project's MIT license does not relicense
those external SDKs, libraries, datasets or checkpoints. This notice is scoped
to the identified source contributions and build dependencies; it makes no
unsupported derivation claim for other algorithms or kernels.

## pybind11 v2.12.0

The Python extension uses pybind11. Its upstream license is retained verbatim below.
Source: https://github.com/pybind/pybind11/blob/v2.12.0/LICENSE

```text
Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
external contributions to this project including patches, pull requests, etc.
```
