# Install dependencies

```
pip install .
```

# Test demo

```
python ./test_grip.py
```

# Dual interactive position demo

One drag bar controls both grippers. Defaults:

- Left hand: `192.168.10.10:55551`
- Right hand: `192.168.10.11:55551`

```
python ./dual_interactive_position.py
```

# Build Python according to packages

```
python -m build
```

# Install automatically built Python packages ...

```
pip install ./dist/dm_lingkong_grip_sdk-1.0.0-py3-none-any.whl
```
