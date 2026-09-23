# LOCA-Bench

This directory includes the environment, MCP servers, and seven context-length
task configurations used by the unified adapter in `../adapter.py`.
Task workspaces are generated during reset; there is no additional dataset download.

From the repository root, with Node.js 24 and npm available:

```bash
pip install -r requirements/loca.txt
npm install -g @modelcontextprotocol/server-filesystem @modelcontextprotocol/server-memory
uv tool install --with 'mcp<2' cli-mcp-server
uv tool install pdf-tools-mcp
python main.py --config configs/loca/pos.yaml
```

Ensure `node`, `npm`, `npx`, and the `uv tool` executables are on `PATH`.
Change `loca_environment_length` in the YAML to choose among 8k, 16k, 32k, 64k,
96k, 128k, and 256k. Each length contains 15 task families with five seeds.

The included code comes from https://github.com/hkust-nlp/LOCA-bench at commit
`8b6fac49d9edd92922593e703b74ea255357c3ec`. Local adaptations select the active
Python interpreter for MCP servers and Python execution and use installed
terminal/PDF executables. One helper's f-string quoting is made compatible with
Python 3.11. Task generation and evaluator logic are unchanged.

The upstream MIT license is retained in `LICENSE`. GEM files retain their
Apache-2.0 copyright headers and license. Third-party authors are not the
anonymous PoS contributors.

Tools can execute code and modify files. Use a disposable isolated environment,
not an account containing private files or production credentials.
