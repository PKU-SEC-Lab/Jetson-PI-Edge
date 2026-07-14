from PIL import Image

# Create a 224x224 gray RGB image.
width, height = 224, 224
image = Image.new("RGB", (width, height), color=(128, 140, 168))

# Save as JPG.
file_path = "tools/mtmd/test-224.jpg"
image.save(file_path, "JPEG")

print(f"Image saved to: {file_path}")
