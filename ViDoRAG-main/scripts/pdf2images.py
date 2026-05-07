import os

from pdf2image import convert_from_path
from tqdm import tqdm

datasets = ["ViDoSeek", "SlideVQA"]

for dataset in datasets:
    root_path = f"./data/{dataset}"
    pdf_path = os.path.join(root_path, "pdf")
    pdf_files = [file for file in os.listdir(pdf_path) if file.endswith("pdf")]
    img_path = os.path.join(root_path, "img")
    os.makedirs(img_path, exist_ok=True)
    for filename in tqdm(pdf_files):
        filepath = os.path.join(pdf_path, filename)
        imgname = filename.split(".pdf")[0]
        images = convert_from_path(filepath, thread_count=4)
        for i, image in enumerate(images):
            idx = i + 1
            image.save(
                os.path.join(root_path, "img", f"{imgname}_{idx}.jpg"), "JPEG"
            )
