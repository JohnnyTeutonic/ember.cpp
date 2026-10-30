# CPU publication integration check

The clean publication checkout builds in Release with CUDA disabled. Its two CTests pass. Fixed UTF-8 cases and 10,000 random plain and streamed cases match the independent Python reference; IR numerical edge cases pass. The current Keramos adapter run records554 passes,0 failures and44 explicit skips (33 unsupported ops,11 gradients). These checks validate the published CPU source; they are not new CUDA performance or conformance evidence.

The build began at586e72c. C++/header/CMake files are unchanged through the source commit recorded in summary.json; later integration changes were documentation, licensing and experiment-driver files. Binary/source hashes, compiler version, raw logs and per-case results are retained here. Models and executables remain in the original build/artifact directories rather than being duplicated in Git.
