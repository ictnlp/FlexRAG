from pathlib import Path

from flexrag.common import configure

from .document_parser_base import DOCUMENTPARSERS, Document, DocumentParserBase


@configure
class DoclingConfig:
    do_ocr: bool = False
    do_table_structure: bool = True
    generate_page_images: bool = False
    generate_picture_images: bool = False


@DOCUMENTPARSERS("docling", config_class=DoclingConfig)
class DoclingParser(DocumentParserBase):
    def __init__(self, config: DoclingConfig):
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as error:
            raise ImportError(
                "Docling is not installed. Install `flexrag[doc-parsers]` or "
                "`docling` to use DoclingParser."
            ) from error

        self.generate_page_images = config.generate_page_images
        self.generate_picture_images = config.generate_picture_images
        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=config.do_ocr,
            do_table_structure=config.do_table_structure,
            generate_page_images=config.generate_page_images,
            generate_picture_images=config.generate_picture_images,
        )
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
            }
        )
        return

    def parse(self, input_file_path: str) -> Document:
        input_path = Path(input_file_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        document_ = self.doc_converter.convert(input_path).document
        document = Document(
            source_file_path=str(input_path),
            text=document_.export_to_markdown(),
            title=document_.name,
        )
        if self.generate_page_images:
            for page_no in sorted(document_.pages):
                image_ref = document_.pages[page_no].image
                if image_ref is not None and image_ref.pil_image is not None:
                    document.screenshots.append(image_ref.pil_image)
        if self.generate_picture_images:
            for picture in document_.pictures:
                image_ref = picture.image
                if image_ref is not None and image_ref.pil_image is not None:
                    document.images.append(image_ref.pil_image)
        return document
