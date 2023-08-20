import stable_diffusion
import upscaler.esrgan

def generate_image(model_path, prompt, use_gpu=True):
  image = stable_diffusion.generate_image(model_path, prompt, use_gpu)
  return image

def upscale_image(image, factor=4):
  upscaled_image = upscaler.esrgan.upscale(image, factor)
  return upscaled_image

def main():
  model_path = "models/testing.ckpt"

  # Get the prompt from the user.
  prompt = input("Enter a prompt: ")

  image = generate_image(model_path, prompt, use_gpu=True)
  upscaled_image = upscale_image(image, factor=4)
  print(upscaled_image)

if __name__ == "__main__":
  main()
