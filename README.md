# Wiseacre

A private AI chat assistant that runs on your own computer. Your conversations
never leave your machine, there is no account, and after the first download it
works without an internet connection.

It picks a Qwen model that fits your computer, opens a chat page in your
browser, and uses speculative decoding (a small helper model drafts text that
the main model checks) to make answers appear faster without changing them.

## Start it

You need Python 3.10 or newer and about 5 to 70 GB of free disk space,
depending on the model chosen for your computer.

**Mac:** double-click `start.command`. If macOS blocks it, right-click the file,
choose Open, then Open again. You can also run `./start.command` in Terminal.

**Windows:** double-click `start.bat`.

**Linux:** run `./start.command` in a terminal.

The first run sets itself up and downloads the model, which can take several
minutes. After that it starts in under a minute and your browser opens to the
chat page. Close the window, or press Ctrl+C, to stop it.

Python is free from python.org. On Windows, tick "Add python.exe to PATH" in
the installer.

## What model will I get?

It checks your memory and graphics hardware and picks automatically. See the
choices for your computer with `./start.command --list` (Windows: `start.bat --list`).

| Tier | Model | Memory needed | Download |
|---|---|---|---|
| xl | Qwen2.5 32B | 90 GB | 66 GB |
| l | Qwen2.5 14B | 40 GB | 30 GB |
| m | Qwen2.5 7B | 20 GB | 15 GB |
| s | Qwen2.5 3B | 12 GB | 6 GB |
| xs | Qwen2.5 1.5B | any | 3 GB |

"Memory" means graphics memory on an NVIDIA GPU, otherwise the computer's RAM
(Apple Silicon Macs share one pool). Computers with no GPU are capped at the
two smallest tiers because larger models are too slow there. Bigger models
give noticeably better answers, so more memory means better quality.

Force a tier with `--tier m`, or use any other model:

```
./start.command --target Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct
```

The draft model must share a tokenizer with the main model. Qwen models of
different sizes do.

## Using it

Type a question and press Enter. Under each answer you will see the speed in
tokens per second. Under the message box, open **Answer options** to change
the instructions given to the assistant, the maximum answer length, how
creative it is, and whether speculative decoding is on. At creativity 0 the
same question gives the same answer every time.

Answers stop at the maximum length (1024 tokens by default, roughly 700
words), sometimes in the middle of a sentence or a code listing. When that
happens the speed line says so. Type "continue" to keep going, or raise the
limit in Answer options. Long requests, such as many homework problems at
once, work better in smaller batches.

## What it remembers

Within one conversation it remembers everything you said earlier, up to about
6000 tokens; in a very long chat the oldest messages are dropped. When you
close the page or click **New chat**, the conversation is forgotten, and the
model itself never learns from your chats.

Two things can be kept on your computer, in a folder called `Wiseacre` inside
your home folder (change it with `--data-dir`):

- **Answer options** are remembered automatically: the instructions for the
  assistant, answer length, creativity and the speculative decoding switch. A
  standing instruction such as "I teach C++ to second-year students" in the
  instructions box is used every time you start Wiseacre.
- **Saved conversations** are off by default. Tick **Save my conversations on
  this computer** under Saved conversations to turn them on. Each conversation
  is then saved as two files in `Wiseacre/chats`: a `.md` file you can read in
  any text editor, and a `.json` copy that Wiseacre reopens. Pick one from
  **Reopen a saved conversation** and click Open to continue it. Delete the
  files to remove a conversation.

## Options

| Option | Effect |
|---|---|
| `--list` | Show the tiers and which one your computer gets |
| `--tier m` | Pick a tier yourself |
| `--no-speculative` | Skip the helper model (uses less memory) |
| `--lan` | Let other computers on your network open the chat. Off by default; anyone on the network could then use it |
| `--port 8000` | Use a different port |
| `--data-dir PATH` | Keep remembered options and saved conversations somewhere other than `~/Wiseacre` |
| `--no-browser` | Do not open the browser automatically |

## Privacy

The chat page is served only to your own computer unless you use `--lan`.
Models are downloaded once from Hugging Face and stored in
`~/.cache/huggingface` (Windows: `C:\Users\you\.cache\huggingface`). Nothing
you type is sent anywhere. To block all network use after the first
download, set the environment variable `HF_HUB_OFFLINE=1`.

## Adding a logo

Save a PNG called `logo.png` in the `assets` folder (create the folder if it is
missing). It then appears beside the page title and as the browser tab icon.
A square image with a transparent background works best.

## How the code is organized

| File | What it does |
|---|---|
| `wiseacre.py` | Starts everything: reads the options, picks the models, opens the page |
| `models.py` | Detects the hardware and holds the table of model tiers (edit it to try other models) |
| `engine.py` | Loads the models and generates answers, with speculative decoding |
| `store.py` | Remembered options and saved conversations |
| `ui.py` | The chat page |
| `tests/smoke_test.py` | Offline checks using tiny stand-in models |

## Licenses

The code in this folder is MIT licensed (see `LICENSE`). The models are not
included; they are downloaded from Hugging Face. Qwen2.5 1.5B, 7B, 14B, 32B
and 0.5B use the Apache 2.0 license. Qwen2.5 3B uses the Qwen Research License,
which is for non-commercial use, and the program prints a notice when it is
chosen. Check the model card on Hugging Face before using any model for
something other than teaching and research.

## Troubleshooting

- **"Python was not found"**: install Python 3.10 or newer, then run the start file again.
- **Out of memory or the computer freezes**: close other programs, or use a smaller tier, for example `--tier s`.
- **Very slow answers**: check the first line it prints. "CPU only" means no GPU was found, and answers will be slow on any model.
- **NVIDIA GPU on Windows not used**: the default PyTorch install is CPU only on Windows. Install the CUDA build from pytorch.org (choose Pip and your CUDA version), inside the `.venv` folder's environment.
- **Starting over**: delete the `.venv` folder and run the start file again.
- **Check the install works**: `python tests/smoke_test.py` runs a quick test with tiny models and no download.
