import logging
import os
from dataclasses import dataclass, field

import gradio as gr
import hydra
import PIL
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from kylin.assistant import Assistant, AssistantConfig
from kylin.searchers import SearcherConfig, load_searcher

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class Config(SearcherConfig):
    assistant_config: AssistantConfig = field(default_factory=AssistantConfig)


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logging.info(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load searcher
    searcher = load_searcher(config)

    # load assistant
    assistant = Assistant(config.assistant_config)

    # launch the gradio app
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "TestLogo.webp")
    image = PIL.Image.open(logo_path)
    theme = gr.themes.Soft()
    with gr.Blocks(
        theme=theme,
        title="Librarian: An AI based search engine framework.",
    ) as demo:
        logo_pic = gr.Image(
            value=image,
            type="pil",
            width="50%",
            show_label=False,
            show_download_button=False,
            show_share_button=False,
            interactive=False,
            container=True,
        )
        with gr.Row(visible=False) as output_row:
            chatbot = gr.Chatbot(
                type="messages",
                label="History messages",
                show_copy_button=True,
            )
            contexts = gr.Chatbot(
                type="messages",
                label="Searched information",
                show_copy_button=True,
            )
        msg = gr.Textbox(
            visible=True, info="What would you like to know?", show_label=False
        )

        def rag_chat(message: str, history: list[dict[str, str]]) -> dict:
            ctxs, _ = searcher.search(message)
            response = assistant.answer(questions=[message], contexts=[ctxs])[0][0]
            history.append(gr.ChatMessage(role="user", content=message))
            history.append(gr.ChatMessage(role="assistant", content=response))
            ctxs = [
                gr.ChatMessage(role="assistant", content=ctx.full_text) for ctx in ctxs
            ]
            return {
                logo_pic: gr.Image(value=image, visible=False),
                output_row: gr.Row(visible=True),
                chatbot: history,
                msg: "",
                contexts: ctxs,
            }

        msg.submit(
            rag_chat,
            inputs=[msg, chatbot],
            outputs=[logo_pic, output_row, msg, chatbot, contexts],
        )

    demo.launch()
    searcher.close()
    return


if __name__ == "__main__":
    main()
