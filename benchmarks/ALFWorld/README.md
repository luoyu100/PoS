# ALFWorld

The unified integration is in `../adapter.py`; the official environment is
installed as the pinned `alfworld` Python package rather than copied here.

From the repository root:

```bash
pip install -r requirements/alfworld.txt
python benchmarks/ALFWorld/download.py
python main.py --config configs/alfworld/pos.yaml
```

The download command installs the 134 valid-unseen text games into
`json_2.1.1/valid_unseen/` and the matching domain files into `logic/`.
To import an existing ALFWorld data directory:

```bash
python benchmarks/ALFWorld/download.py --source /path/to/alfworld-data
```

Source: https://github.com/alfworld/alfworld. The official package and data retain
their upstream licenses. Linux x86_64 is recommended for the TextWorld dependency.
