import glob
import json
import os.path
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def parse_arguements() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing the Fashionpedia data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguements()
    image_paths = glob.glob(os.path.join(args.data_dir, "**/*.jpg"))
    num_chunks = int(np.ceil(len(image_paths) / args.batch_size))
    embedding_model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(embedding_model_id).eval()
    processor = CLIPProcessor.from_pretrained(embedding_model_id)
    result = {
        "dataset": "fashionpedia",
        "embedding_model_id": embedding_model_id,
    }
    embeddings_by_path = {}
    for chunk_idx in tqdm.tqdm(range(num_chunks)):
        start_idx = chunk_idx * args.batch_size
        end_idx = (chunk_idx + 1) * args.batch_size

        image_paths_batch = image_paths[start_idx:end_idx]
        images = [Image.open(path).convert("RGB") for path in image_paths_batch]
        inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().detach().tolist()
        embeddings_by_path = {
            **embeddings_by_path,
            **{
                path: embedding
                for path, embedding in zip(image_paths_batch, embeddings)
            },
        }
    result["embeddings"] = embeddings_by_path
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fashionpedia-embeddings.json"), "w") as f:
        json.dump(result, f)
