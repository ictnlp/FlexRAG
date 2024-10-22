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
        if searcher is not None:
            with gr.Row(visible=False) as output_row:
                chatbot = gr.Chatbot(
                    type="messages",
                    label="History messages",
                    show_copy_button=True,
                )
                context_box = gr.Chatbot(
                    type="messages",
                    label="Searched information",
                    show_copy_button=True,
                    visible=searcher is not None,
                )
        else:
            chatbot = gr.Chatbot(
                type="messages",
                label="History messages",
                show_copy_button=True,
                visible=False,
            )
        msg = gr.Textbox(
            visible=True, info="What would you like to know?", show_label=False
        )

        def rag_chat(message: str, history: list[dict[str, str]]) -> dict:
            if searcher is not None:
                ctxs = searcher.search(message)[0]
                response = assistant.answer(
                    questions=[message], contexts=[ctxs], histories=[history]
                )[0][0]
            else:
                ctxs = []
                response = assistant.answer(
                    questions=[message], histories=[history]
                )[0][0]  # fmt: skip
            history.append(gr.ChatMessage(role="user", content=message))
            history.append(gr.ChatMessage(role="assistant", content=response))

            if searcher is not None:
                ctxs = [
                    gr.ChatMessage(role="assistant", content=ctx.full_text)
                    for ctx in ctxs
                ]
                r = {
                    logo_pic: gr.Image(value=image, visible=False),
                    output_row: gr.Row(visible=True),
                    chatbot: history,
                    msg: "",
                    context_box: ctxs,
                }
            else:
                r = {
                    logo_pic: gr.Image(value=image, visible=False),
                    chatbot: gr.Chatbot(
                        type="messages",
                        label="History messages",
                        show_copy_button=True,
                        visible=True,
                    ),
                    chatbot: history,
                    msg: "",
                }
            return r

        if searcher is not None:
            outputs = [logo_pic, output_row, chatbot, msg, context_box]
        else:
            outputs = [logo_pic, chatbot, chatbot, msg]
        msg.submit(
            rag_chat,
            inputs=[msg, chatbot],
            outputs=outputs,
        )

    demo.launch()
    searcher.close()
    return


if __name__ == "__main__":
    main()
