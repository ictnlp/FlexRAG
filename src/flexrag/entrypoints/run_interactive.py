import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import PIL.Image
import gradio as gr
import hydra
import PIL
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from flexrag.assistant import ASSISTANTS
from flexrag.utils import load_user_module, LOGGER_MANAGER


# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


AssistantConfig = ASSISTANTS.make_config()


@dataclass
class Config(AssistantConfig):
    share: bool = False
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    auth: Optional[list[str]] = None


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


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)

    # load assistant
    assistant = ASSISTANTS.load(config)

    # launch the gradio app
    logo = PIL.Image.open(logo_path)
    wide_logo = PIL.Image.open(wide_logo_path)
    theme = gr.themes.Soft()
    with gr.Blocks(
        theme=theme,
        title="📖flexrag: A RAG Framework for Information Retrieval and Generation.",
        fill_height=True,
        css=custom_css,
    ) as demo:
        logo_pic = gr.Image(
            value=logo,
            image_mode="RGBA",
            type="pil",
            width="40%",
            show_label=False,
            show_download_button=False,
            show_share_button=False,
            show_fullscreen_button=False,
            interactive=False,
            container=True,
            elem_id="logo",
        )
        with gr.Row(visible=False, max_height="100%") as output_row:
            chatbot = gr.Chatbot(
                type="messages",
                label="History messages",
                show_copy_button=True,
                height="100%",
                max_height="100%",
                avatar_images=[robot_path, user_path],
            )
            context_box = gr.Chatbot(
                type="messages",
                label="Searched contexts",
                show_copy_button=True,
                visible=assistant is not None,
                height="100%",
                max_height="100%",
            )
        msg = gr.Textbox(
            visible=True,
            info="What would you like to know?",
            show_label=False,
            submit_btn=True,
            stop_btn=False,
        )
        clear_btn = gr.ClearButton([msg, chatbot, context_box], visible=False)

        def rag_chat(message: str, history: list[dict[str, str]]) -> dict:
            response, contexts, _ = assistant.answer(question=message)
            history.append(gr.ChatMessage(role="user", content=message))
            history.append(gr.ChatMessage(role="assistant", content=response))

            ctxs = [
                gr.ChatMessage(
                    role="assistant",
                    content=ctx.data["text"],
                    metadata={"title": f"Retrieved by: {ctx.retriever}"},
                )
                for ctx in contexts
            ]
            r = {
                logo_pic: wide_logo,
                output_row: gr.Row(
                    visible=True,
                    height=720,
                ),
                chatbot: history,
                msg: "",
                context_box: ctxs,
                clear_btn: gr.ClearButton([msg, chatbot, context_box], visible=True),
            }
            return r

        msg.submit(
            rag_chat,
            inputs=[msg, chatbot],
            outputs=[logo_pic, output_row, chatbot, msg, context_box, clear_btn],
        )

    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share,
        auth=config.auth,
    )
    return


if __name__ == "__main__":
    main()
