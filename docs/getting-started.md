# Getting started

The step-by-step version of the first-steps demo in the README.

## 1. Clone

```bash
git clone https://github.com/alainnothere/privibe ~/privibe
```

## 2. Install uv

The standalone installer is the recommended path (it also puts `uv` on your
PATH):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or through pip if you prefer:

```bash
pip install --user uv --break-system-packages
```

Note: `--break-system-packages` bypasses your distro's protection for the
system Python; the standalone installer avoids the question entirely. If `uv`
is not on your PATH after a pip install, call it as `~/.local/bin/uv`.

## 3. Have a model server running

privibe's default config already points at a local llama.cpp server on
`http://127.0.0.1:8080/v1`, so if your server runs there you have nothing to
configure. Any OpenAI-compatible server works. A minimal llama-server example:

```bash
llama-server -m your-model.gguf -c 32768 --port 8080
```

For per-message reasoning effort (`/effort`) and disk KV-cache eviction, use
the companion llama-server build linked in the README.

## 4. First run

```bash
cd ~/privibe
uv run privibe
```

The first run writes the config to `~/.privibe/config.toml`.

If your shell exports a proxy, blank it for the run so localhost traffic does
not get routed through it:

```bash
http_proxy= uv run privibe
```

## 5. Context files and extra instructions

Context files let you reuse "knowledge" gathered in previous conversations
without redoing the whole research. Create a folder for them, and an
instructions file that tells privibe where they live:

```bash
mkdir ~/contextFiles
echo 'context files folder is ~/contextFiles' > ~/privibeInstructions.md
```

## 6. Tune the config

Edit `~/.privibe/config.toml`:

```toml
# force the hashed editing tools; they are more reliable with smaller local models
disabled_tools = [
    "search_replace", "read_file", "write_file", "ask_user_question", "todo", "task",
]

extra_instruction_files = [
    "~/privibeInstructions.md",
]
```

`~` works in these paths and expands to your home directory.

## 7. Say hi

```bash
cd ~/privibe
uv run privibe
```

You are ready to work. A good first ask: point it at a project and have it
write a readme.

## Windows

Same flow from Git Bash; see
[running-in-windows-git-bash.md](running-in-windows-git-bash.md).
