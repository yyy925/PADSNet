# Split files

Create `train.txt` and `test.txt` in this directory for local use. Each row has the form:

```text
relative/sample/directory integer_label
```

Example:

```text
high-risk/sample_001 0
low-risk/sample_002 1
normal/sample_003 2
```

The paths are interpreted relative to `data/frames/train` or `data/frames/test`. Real patient identifiers must not be committed.
