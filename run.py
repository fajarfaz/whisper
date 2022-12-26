import whisper
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("base", device = device )
result = model.transcribe("audio.m4a", fp16=False)
print(result["text"])
