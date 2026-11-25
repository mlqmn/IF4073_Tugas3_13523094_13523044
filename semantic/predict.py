import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from model import load_trained_model, predict_class_map

USAGE = "Penggunaan: python predict.py <citra_masukan> <peta_label_keluaran.png>"


def main():
    if len(sys.argv) != 3:
        sys.exit(USAGE)
    image_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])

    try:
        image = Image.open(image_path)
    except (FileNotFoundError, UnidentifiedImageError):
        sys.exit(f"Citra tidak dapat dibaca: {image_path}")

    try:
        model = load_trained_model()
    except FileNotFoundError as error:
        sys.exit(str(error))

    Image.fromarray(predict_class_map(model, image)).save(output_path)


if __name__ == "__main__":
    main()
