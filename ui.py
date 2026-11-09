"""The chat page: a Gradio app that talks to the Engine and the Store."""

import base64
import html
import re
from pathlib import Path

import gradio as gr

from store import DEFAULT_SETTINGS

ASSETS = Path(__file__).resolve().parent / "assets"
STATS_MARK = "⏱"   # stopwatch; marks the speed line added under each answer


# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------

def stats_line(stats):
    """The small italic line under an answer: speed, and a note if it was cut off."""
    mode = "on" if stats.speculative else "off"
    line = (f"{STATS_MARK} {stats.tokens} tokens in {stats.seconds:.1f} s "
            f"({stats.tokens_per_second:.1f} tokens/s), speculative decoding {mode}.")
    if stats.truncated:
        line += (" **This answer was cut off at the length limit.** "
                 "Type “continue” to keep going, or raise the limit under Answer options.")
    return f"\n\n*{line}*"


_STATS_LINE = re.compile(rf"\n\n\*{STATS_MARK}.*\*\s*$", re.S)


def clean_history(history):
    """Remove the speed line from past answers before they go back to the model."""
    cleaned = []
    for message in history:
        content = message["content"]
        if isinstance(content, list):  # multimodal-style content
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if message["role"] == "assistant":
            content = _STATS_LINE.sub("", content)
        cleaned.append({"role": message["role"], "content": content})
    return cleaned


def header_html(title):
    """The page title, with a logo beside it if assets/logo.png exists."""
    logo = ""
    try:
        data = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()
        logo = f'<img src="data:image/png;base64,{data}" alt="" style="height:56px;width:auto">'
    except OSError:
        pass  # no logo file: just the title
    return ('<div style="display:flex;align-items:center;justify-content:center;gap:14px;margin:4px 0">'
            f'{logo}<h1 style="margin:0">{html.escape(title)}</h1></div>')


# ------------------------------------------------------------
# The app
# ------------------------------------------------------------

class ChatApp:
    def __init__(self, engine, store):
        self.engine = engine
        self.store = store

    # -- what the buttons do ---------------------------------------------

    def respond(self, message, history, system_prompt, max_new, temperature, use_spec, save_on, current):
        """Answer one message, streaming the text into the chat as it is written.
        `current` is the file name of the conversation being saved, if any."""
        message = (message or "").strip()
        history = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        if not message:
            yield history, "", current, gr.update()
            return

        history.append({"role": "user", "content": message})
        yield history + [{"role": "assistant", "content": "Thinking…"}], "", current, gr.update()

        shown = ""
        try:
            for text, stats in self.engine.stream(
                    clean_history(history), system_prompt, int(max_new), float(temperature), bool(use_spec)):
                shown = text if stats is None else text + stats_line(stats)
                yield history + [{"role": "assistant", "content": shown}], "", current, gr.update()
        except Exception as e:  # show the problem in the chat instead of a blank page
            shown = (shown + "\n\n" if shown else "") + f"*Something went wrong: {e}*"
        history.append({"role": "assistant", "content": shown})

        picker = gr.update()
        if save_on:
            try:
                current = self.store.save_chat(current, clean_history(history), self.engine.target_name)
                picker = gr.update(choices=self.store.list_chats(), value=current)
            except OSError as e:
                history[-1]["content"] += f"\n\n*Could not save this conversation: {e}*"
        yield history, "", current, picker

    def persist(self, *values):
        """Remember the options for next time. Values arrive in DEFAULT_SETTINGS order."""
        self.store.save_settings({key: type(DEFAULT_SETTINGS[key])(value)
                                  for key, value in zip(DEFAULT_SETTINGS, values)})

    def open_chat(self, filename):
        messages = self.store.load_chat(filename)
        if messages is None:
            gr.Warning("That conversation could not be opened.")
            return gr.update(), gr.update()
        return messages, filename

    def refresh_list(self):
        return gr.update(choices=self.store.list_chats())

    # -- the page ---------------------------------------------------------

    def build(self, title):
        settings = self.store.load_settings()
        with gr.Blocks(title=title, css="footer {display: none !important}") as demo:
            gr.HTML(header_html(title))
            gr.Markdown("Runs entirely on this computer. Your conversation is not sent anywhere. "
                        "Answers can be wrong, so check anything important.")
            chatbot = gr.Chatbot(type="messages", height="55vh", show_copy_button=True, label="Conversation")
            with gr.Row():
                box = gr.Textbox(placeholder="Type a message and press Enter", show_label=False,
                                 scale=8, lines=1, max_lines=6, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1, min_width=80)
                new = gr.Button("New chat", scale=1, min_width=80)

            answer_options = self._options_panel(settings)
            save_box, picker, open_btn, refresh_btn = self._saved_chats_panel(settings)
            options = [*answer_options, save_box]   # the same order as DEFAULT_SETTINGS
            current = gr.State(None)  # file name of the conversation being saved

            run = dict(fn=self.respond, inputs=[box, chatbot, *options, current],
                       outputs=[chatbot, box, current, picker], concurrency_limit=1)
            box.submit(api_name="chat", **run)
            send.click(api_name=False, **run)
            new.click(lambda: ([], "", None), None, [chatbot, box, current], api_name=False)
            open_btn.click(self.open_chat, picker, [chatbot, current], api_name=False)
            refresh_btn.click(self.refresh_list, None, picker, api_name=False)
            demo.load(self.refresh_list, None, picker, api_name=False)

            # Remember options as they change. The text box saves when you click away,
            # not on every keystroke.
            for component in options:
                event = component.blur if isinstance(component, gr.Textbox) else component.change
                event(self.persist, options, None, show_progress="hidden", api_name=False)
        return demo

    def _options_panel(self, settings):
        """The 'Answer options' panel. Returns its four components, in DEFAULT_SETTINGS order."""
        with gr.Accordion("Answer options: length, creativity, instructions", open=False):
            system_box = gr.Textbox(settings["system_prompt"], label="Instructions for the assistant", lines=2)
            max_slider = gr.Slider(64, 4096, value=settings["max_new"], step=64,
                                   label="Maximum answer length (tokens; about 3 tokens per 2 words)")
            temp_slider = gr.Slider(0.0, 1.2, value=settings["temperature"], step=0.1,
                                    label="Creativity (0 = most predictable and repeatable)")
            can_speculate = self.engine.speculative_ok
            spec_box = gr.Checkbox(settings["speculative"] and can_speculate,
                                   label="Speculative decoding (faster answers, same quality)",
                                   interactive=can_speculate)
            gr.Markdown("These options are remembered the next time you start Wiseacre.")
        return [system_box, max_slider, temp_slider, spec_box]

    def _saved_chats_panel(self, settings):
        with gr.Accordion("Saved conversations", open=False):
            save_box = gr.Checkbox(settings["save_chats"], label="Save my conversations on this computer")
            gr.Markdown(f"Off by default. When on, each conversation is saved in `{self.store.chat_dir}`, "
                        "as a file you can read and a copy this page can reopen. "
                        "Delete the files there to remove a conversation.")
            with gr.Row():
                picker = gr.Dropdown(choices=self.store.list_chats(), label="Reopen a saved conversation", scale=4)
                open_btn = gr.Button("Open", scale=1, min_width=80)
                refresh_btn = gr.Button("Refresh list", scale=1, min_width=80)
        return save_box, picker, open_btn, refresh_btn


def build_ui(engine, title, store):
    return ChatApp(engine, store).build(title)
