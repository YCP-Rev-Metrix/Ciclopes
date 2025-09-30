import torch

def list_gpus():
    if not torch.cuda.is_available():
        print("No GPU")
        return
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    list_gpus()