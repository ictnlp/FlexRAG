import sys
from pathlib import Path
from typing import Callable, Optional

import hydra
import PIL
import PIL.Image
from hydra.core.config_store import ConfigStore

try:
    import gradio as gr
except ImportError as error:
    raise ImportError(
        "Gradio is not installed. Install `flexrag[ui]` or `gradio` to use "
        "the interactive UI entrypoint."
    ) from error

from flexrag.assistants import ASSISTANTS
from flexrag.common import LOGGER_MANAGER, configure, extract_config, load_user_module
from flexrag.common.dataclasses import ChatMessages, ChatTurn

# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


AssistantConfig = ASSISTANTS.make_config(config_name="AssistantConfig")


@configure
class Config(AssistantConfig):
    share: bool = False
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    auth: Optional[list[str]] = None
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("run_interactive")


# prepare resources
custom_css = """
#logo {
    background-color: transparent;    
}
"""
logo_path = Path(__file__).parents[0] / "assets" / "flexrag.png"
wide_logo_path = Path(__file__).parents[0] / "assets" / "flexrag-wide.png"
robot_path = Path(__file__).parents[0] / "assets" / "robot.png"
user_path = Path(__file__).parents[0] / "assets" / "user.png"


def app(
    chat_func: Callable[[str, list[dict]], list[gr.ChatMessage]],
    server_name: str,
    server_port: int,
    share: bool,
    auth: Optional[list[str]],
    debug: bool,
):
    wide_logo = PIL.Image.open(wide_logo_path)
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="📖flexrag: A RAG Framework for Information Retrieval and Generation.",
        fill_height=True,
        fill_width=False,
        css=custom_css,
    ) as demo:
        with gr.Column():
            logo_pic = gr.Image(
                value=wide_logo,
                image_mode="RGBA",
                type="pil",
                width="60%",
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                show_fullscreen_button=False,
                interactive=False,
                container=True,
                elem_id="logo",
            )
            chatbot = gr.Chatbot(
                type="messages",
                avatar_images=[user_path, robot_path],
                height="80%",
                elem_id="chatbot",
            )
            chatbox = gr.ChatInterface(
                chat_func,
                type="messages",
                chatbot=chatbot,
                flagging_mode="manual",
                flagging_options=[
                    "Like",
                    "Irrelevant Context",
                    "Unfaithful to Context",
                ],
                save_history=True,
                editable=True,
                theme=gr.themes.Soft(),
                fill_height=True,
                fill_width=False,
                # multimodal=True,  # This option is not compatible with `editable=True`
            )
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth=auth,
        debug=debug,
    )
    return


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    config = extract_config(config, Config)
    logger.debug(f"Configs:\n{config.dumps()}")

    # load assistant
    assistant = ASSISTANTS.load(config)

    def rag_chat(message: str, history: list[dict[str, str]]) -> list[gr.ChatMessage]:
        history_ = []
        for turn in history:
            if turn["role"] == "assistant":
                if len(turn["metadata"]) > 0:
                    continue
            history_.append(turn)
        messages = ChatMessages.from_list(history_)
        messages.append(ChatTurn(role="user", content=message))
        response = assistant.answer(messages)

        r = []
        # add contexts to the response messages
        if response.contexts is not None:
            for ctx in response.contexts:
                r.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=ctx.data["text"],
                        metadata={"title": f"Retrieved by: {ctx.retriever}"},
                    )
                )
        # add the final response
        r.append(
            gr.ChatMessage(
                role="assistant",
                content=response.response,
            )
        )
        return r

    app(
        chat_func=rag_chat,
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share,
        auth=config.auth,
        debug=config.debug,
    )
    return


if __name__ == "__main__":
    main()
