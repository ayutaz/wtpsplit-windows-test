# wtpsplit run for windows

[wtpsplit](https://github.com/segment-any-text/wtpsplit)をWindowsで動かすテスト

# setup

```bash
uv venv -p 3.10
.venv/bin/activate
```

```bash
uv pip install wtpsplit==2.1.4 numpy==1.26.4
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

# run

```bash
python sample.py
```