import librosa
import IPython.display as ipd
import torch
import numpy as np
from src.model1 import AudioEncoder, AudioDecoder, ConvTransformerAudioClassifier

# Instrument mapping (indexing starts from 1)
Reduced_inst_map = {
    1: "Piano",
    2: "Organ",
    3: "Guitar",
    4: "Bass",
    5: "Strings",
    6: "Brass",
    7: "Reed",
    8: "Pipe",
    9: "Synth Lead",
    10: "Other"
}


def predict_instruments(audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load the model
    model = ConvTransformerAudioClassifier()
    model.load_state_dict(torch.load(
        "../models/my_model.pth", map_location=device))
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100)

    # show audio
    ipd.Audio(audio, rate=sr)

    # Chunk size = 10 seconds
    chunk_size = 10 * sr
    num_chunks = len(audio) // chunk_size

    # Only keep full-length chunks
    chunks = np.array_split(audio[:num_chunks * chunk_size], num_chunks)

    # Track predictions
    predicted_indices = []

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            # Convert to tensor and shape appropriately (add batch/channel dims as needed)
            x = torch.tensor(chunk, dtype=torch.float32).to(device)
            # (B, C, T) assuming mono waveform input
            x = x.unsqueeze(0)

            preds = model(x)
            # +1 to match dictionary keys
            pred_index = torch.argmax(preds, dim=1).item() + 1
            predicted_indices.append(pred_index)

    # Get unique predicted instruments
    unique_preds = sorted(set(predicted_indices))
    instruments = [Reduced_inst_map[idx] for idx in unique_preds]

    # Print result
    print("This audio was detected to have the following instruments:\n")
    for inst in instruments:
        print(f"- {inst}")
