import stable_diffusion
import upscaler.esrgan

def generate_image(model_path, prompt, use_gpu=True):
  image = stable_diffusion.generate_image(model_path, prompt, use_gpu)
  return image

def upscale_image(image, factor=4,base_size=(512, 512)):
  upscaled_image = upscaler.esrgan.upscale(image, factor, base_size)
  return upscaled_image

def save_image(image, filename, quality=98):
  image.save(filename, quality=quality)

def main():
  model_path = "path/to/model.ckpt"
  prompt = input("Enter a prompt: ")

  # Split the prompt into a positive prompt and a negative prompt.
  positive_prompt, negative_prompt = prompt.split("-")

  # Generate the image.
  image = generate_image(model_path, positive_prompt, use_gpu)

  # Upscale the image.
  upscaled_image = upscale_image(image, factor=4, base_size=(512, 512))

  # Save the image.
  save_image(upscaled_image, "upscaled_image.jpg", quality=98)

if __name__ == "__main__":
  main()
