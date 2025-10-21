from datasets import GeneratorBasedBuilder, DatasetInfo, SplitGenerator, Features, Image, Value
import os
import json
from typing import List, Tuple, Dict, Any

class ProcessedImageDataset(GeneratorBasedBuilder):
    """
    A dataset builder for the A/B task where conditioning images are in an adjacent directory.
    It expects 'data_dir' to point to the main image directory (e.g., '.../_omg_img_omg').
    It finds control images in a parallel directory (e.g., '.../omg_seg_omg').
    Prompts are loaded from .txt files that accompany each image.
    """
    VERSION = "1.1.0"

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="Dataset for A/B image-to-image translation with paired images in separate folders.",
            features=self._features(),
            supervised_keys=("image_file", "text"),
        )

    def _features(self):
        """Defines the dataset features. We add 'file_path' to find the control image later."""
        return Features({
            "image_file": Image(),
            "text": Value("string"),
            "file_path": Value("string"),  # Pass the original file path to the transform
        "control_img_path": Value("string"),  # <-- Add this line
        "test": Value("string")             # <-- Add this line
        })

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        """Defines the data splits. It verifies that both image and control directories exist."""
        if self.config.data_dir is None:
            raise ValueError("Data directory must be specified. e.g., load_dataset('path/to/script.py', data_dir='path/to/_omg_img_omg')")

        # Use absolute path to make directory logic more robust
        img_dir = os.path.abspath(self.config.data_dir)
        
        # Automatically determine the control directory path
        parent_dir = os.path.dirname(img_dir)
        img_folder_name = os.path.basename(img_dir)
        control_folder_name = img_folder_name.replace('img', 'seg')
        control_dir = os.path.join(parent_dir, control_folder_name)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"The specified image directory does not exist: {img_dir}")
        if not os.path.isdir(control_dir):
            raise FileNotFoundError(f"Could not find the control image directory. Expected at: {control_dir}")

        return [SplitGenerator(name="train", gen_kwargs={"img_dir": img_dir, "control_dir": control_dir})]

    def _generate_examples(self, img_dir: str, control_dir: str) -> Tuple[int, Dict[str, Any]]:
        """
        Yields examples by pairing images with prompts from .txt files and control images.
        """
        pass
        key = 0
        for filename in sorted(os.listdir(img_dir)):
            # Check if the file is an image
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                base_name, _ = os.path.splitext(filename)
                text_file_path = os.path.join(img_dir, f"{base_name}.txt")
                control_img_path = os.path.join(control_dir, filename)

                # An example is only valid if the image, its .txt prompt, and its control image all exist.
                if os.path.exists(text_file_path) and os.path.exists(control_img_path):
                    # Read the prompt from the text file
                    with open(text_file_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()

                    img_path = os.path.join(img_dir, filename)
                    print ( text_file_path, control_img_path , 'data paths')
                    yield key, {
                        "image_file": img_path,
                        "text": prompt,
                        "file_path": img_path,  # This is the crucial part for the transform
                        "control_img_path": control_img_path,  # This is the crucial part for the transform
                        "test":'test'
                    }
                    key += 1