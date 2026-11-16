# Changelog

Each release is a heading with its tag. The release workflow uses the words under it as the release's notes.

## v1.0.0

- The first release: `build.py` (checkpoint + groups file → static 3D site), `init_groups.py` (a drafted groups
  file for any checkpoint), `capture_values.py` (one real run's stage outputs), `bundle_shared.py` (the shared-view
  version), the page and the embeddable module.
- Reads safetensors (single or sharded, memory-mapped) and PyTorch state dicts; other formats stop with what to convert to.
- The page knows no model: colours and the Input and Output marks come from each group's `role` in the groups file
  (or the role holding most of its weights), never from group or module names.
