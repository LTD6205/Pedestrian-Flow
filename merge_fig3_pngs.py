from pathlib import Path
from PIL import Image


def merge_fig3_pngs(
    img1_path: str = "output/fig3_time_development_rho_1.16.png",
    img2_path: str = "output/fig3_time_development_rho_1.21.png",
    out_path: str = "output/fig3_two_panel_merged.png",
    gap: int = 25,
    outer_margin: int = 10,
) -> str:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # Make both images the same height
    target_height = max(img1.height, img2.height)

    def resize_to_height(img: Image.Image, height: int) -> Image.Image:
        if img.height == height:
            return img
        new_width = int(img.width * height / img.height)
        return img.resize((new_width, height), Image.LANCZOS)

    img1 = resize_to_height(img1, target_height)
    img2 = resize_to_height(img2, target_height)

    total_width = img1.width + img2.width + gap + 2 * outer_margin
    total_height = target_height + 2 * outer_margin

    canvas = Image.new("RGB", (total_width, total_height), "white")

    x1 = outer_margin
    y1 = outer_margin

    x2 = outer_margin + img1.width + gap
    y2 = outer_margin

    canvas.paste(img1, (x1, y1))
    canvas.paste(img2, (x2, y2))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)

    return str(out_file)


if __name__ == "__main__":
    saved = merge_fig3_pngs()
    print("Saved merged figure to:")
    print(saved)