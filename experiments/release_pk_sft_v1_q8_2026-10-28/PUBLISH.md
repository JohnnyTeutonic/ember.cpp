# Publishing pk_sft_v1 F16 to the Space and the site

**Completed 30 October 2026 under the author's autonomous completion mission.** Space commit
`8e107b9706f81fec53749fa59da7602344e6c776`; website commit `fdbaa1deded14881cfbe70c10526c1fe915c2c3d`.
Remote manifest verification is recorded in `published_2026-10-30.json`. Commands below preserve the reproduction procedure.

Staged: WSL `~/e22rel/space` (13 files, 214 MB; `staged/MANIFEST.sha256`), the site page
`staged/chat.html`. Run from Git Bash on the laptop. The token is read by `publish_space.py` from the
one `HF_TOKEN=` line of `website/.env` and never printed; no other line of that file is used.

## 1. The Space (huggingface.co/spaces/johnnyteutonic/ember-chat)

Optional check first (verifies the manifest and that the token's account is johnnyteutonic; uploads
nothing):

    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" python3 /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/release_pk_sft_v1_q8_2026-10-28/publish_space.py --space-dir /home/jonat/e22rel/space --env-file /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/website/.env --confirm ember-chat --dry-run

The upload:

    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" python3 /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/release_pk_sft_v1_q8_2026-10-28/publish_space.py --space-dir /home/jonat/e22rel/space --env-file /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/website/.env --confirm ember-chat

What it changes publicly: one commit to the Space repo. `chat.gguf` goes from chat_sftfinal_v2 Q8_0-all
(14.6M parameters, 19 MB) to pk_sft_v1 F16 (97.5M parameters, 221 MB, sha256 3916b3ae...);
`ember.js`/`ember.wasm` become the ember 9d993f4 build with a 2gb heap ceiling (was d2de77e);
`README.md` (the Space card), `index.html`, `SOURCE.txt`, `chat.js`, `worker.js`, `style.css` are
replaced; `webgpu/` (gguf.js, llama_webgpu.js, ir_webgpu.js) is added, so browsers with WebGPU run the
forward pass on the GPU. Nothing is deleted. Visitors then download 221 MB once and the page uses
about 1.5 GB of memory (desktop browsers; phones may fail to load it). The model is stored on the
Space (Git LFS) and uploaded from the laptop: 224 MB at the 0.3-0.5 MB/s uplink takes about 7.5-12.5
minutes.

## 2. The site (johnnyteutonic.github.io/chat.html), after the Space shows the new model

    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/release_pk_sft_v1_q8_2026-10-28/staged/chat.html /home/jonat/JohnnyTeutonic.github.io/chat.html
    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" git -C /home/jonat/JohnnyTeutonic.github.io add chat.html
    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" git -C /home/jonat/JohnnyTeutonic.github.io commit -m "Chat page: pk_sft_v1 (97.5M parameters, 221 MB 16-bit download), its data, evaluation and licences" -- chat.html
    MSYS_NO_PATHCONV=1 wsl.exe --cd "~" timeout 100 git -C /home/jonat/JohnnyTeutonic.github.io push git@github.com:JohnnyTeutonic/JohnnyTeutonic.github.io.git main

What it changes publicly: GitHub Pages republishes chat.html (a minute or two): the model's size,
architecture, pretraining and fine-tuning data, evaluation (simple-QA 21/60) and licences, the
download size (221 MB, 16-bit; the live page says 71 MB). The iframe URL is unchanged.

The workspace's source copy of the site (local-only root repository, nothing public):

    cp /c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/release_pk_sft_v1_q8_2026-10-28/staged/chat.html /c/Users/jonat/OneDrive/Documents/research_portfolio_complete/website/chat.html
    cd /c/Users/jonat/OneDrive/Documents/research_portfolio_complete && git add website/chat.html && git commit -m "website/chat.html: pk_sft_v1 on the Space" -- website/chat.html

## 3. Closure

- Local `deploy/wasm/README_space.md` and `deploy/wasm/web/index.html` now describe the live Space.
- The release tool's former file-size-only heap heuristic is fixed: the F32 export is the size floor
  for expanded weights (fdeb1e1, 10 offline tests pass). The published build is the prior gated 2gb binary.
- The upload's working model copy is replaced by a link to its canonical artifact after verifying both hashes;
  `C:/ml_artifacts/general_chat/sft/pk_sft_v1_f16.gguf` remains the model's home.
