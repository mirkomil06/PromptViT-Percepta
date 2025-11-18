import torch
import argparse
from PIL import Image
from torchvision import transforms

from src.models.vit_baseline import ViTBaseline


def load_class_names(cub_root):
    """
    Загружает настоящие названия классов из CUB_200_2011/classes.txt
    id (1..200) -> "Black_footed_Albatross", ...
    """
    class_file = f"{cub_root}/classes.txt"
    id_to_name = {}
    with open(class_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split()
            id_to_name[int(class_id) - 1] = class_name  # в датасете 0..199
    return id_to_name


def load_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # shape: (1,3,224,224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image.jpg")
    parser.add_argument("--model", default="outputs/cub_baseline_cpu/best_model.pth")
    parser.add_argument("--cub_root", default="data/cub200/CUB_200_2011")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("[Infer] Running on CPU.")

    # 1. Загрузка модели
    print("[Infer] Loading model...")
    model = ViTBaseline(num_classes=200, pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # 2. Загружаем картинку
    print("[Infer] Loading image...")
    img = load_image(args.image)
    img = img.to(device)

    # 3. Классификация
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

    predicted_idx = top_class.item()
    confidence = top_prob.item()

    # 4. Загружаем названия классов
    class_names = load_class_names(args.cub_root)
    predicted_name = class_names[predicted_idx]

    print("\n==== RESULT ====")
    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted class name : {predicted_name}")
    print(f"Confidence: {confidence:.4f}")
    print("=================\n")


if __name__ == "__main__":
    main()
