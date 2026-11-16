# Changelog

Each release is a heading with its tag. The release workflow uses the words under it as the release's notes.

## v1.0.0

- The first release: `build.py` (checkpoint + groups file → static 3D site), `init_groups.py` (a drafted groups
  file for any checkpoint), `capture_values.py` (one real run's stage outputs), `bundle_shared.py` (the shared-view
  version), the page and the embeddable module.
- Reads safetensors (single or sharded, memory-mapped) and PyTorch state dicts; other formats stop with what to convert to.
- Values that aren't learned (GPT-2's causal mask, position numbers, rotary frequencies, norm statistics) are drawn
  but counted apart: "N learned weights, plus M fixed values". Leading size-1 dims are dropped before drawing, so a
  [1, 1, 1024, 1024] mask is a square, not a line.
- `init_groups.py` puts a final norm in Output, names the mask, drafts the residual `skip` links, and marks an output
  head tied to the word table (from `config.json`).
- The page knows no model: colours and the Input and Output marks come from each group's `role` in the groups file
  (or the role holding most of its weights), never from group or module names.
