# Explicit Linux CI baseline, 3 November 2026

Pin Ember Linux wheel, release and supporting runner jobs to ubuntu-24.04. The wheel matrix OS condition changes with the matrix label so its Linux-only branch continues to execute. Published release artifact/asset names remain unchanged. Windows and macOS rows are unchanged.

This follows the workspace report that ubuntu-latest advanced to a newer compiler/static-analysis environment and Paperkiln PR7 pinned its known baseline. It does not establish GCC15 incompatibility in Ember or claim that hosted jobs have passed; it makes the tested host baseline explicit. Both workflow documents parse as YAML and runner/matrix conditions were checked. Keramos currently has no workflow directory in its workspace checkout, so no workflow was invented or changed there. Hosted Actions results remain separate evidence.
