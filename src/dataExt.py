import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

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
reduced_instruments_map = {
    1: [1, 2, 3, 4, 5, 6, 7, 8],
    2: [17, 18, 19, 20, 21, 22, 23, 24],
    3: [25, 26, 27, 28, 29, 30, 31, 32],
    4: [33, 34, 35, 36, 37, 38, 39, 40],
    5: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
    6: [57, 58, 59, 60, 61, 62, 63, 64],
    7: [65, 66, 67, 68, 69, 70, 71, 72],
    8: [73, 74, 75, 76, 77, 78, 79, 80],
    9: [81, 82, 83, 84, 85, 86, 87, 88],
    10: [9, 10, 11, 12, 13, 14, 15, 16, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
}


class MusicNetDataset(Dataset):
    def __init__(self, parent, sr=44100):
        self.parent = parent
        self.chunk_size = 10 * sr
        self.data = []

        for id in parent.ids:
            audio, sr, labels = parent.access(id)
            if sr != 44100:
                print(f"Resampling {id} from {sr} to 44100")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
                sr = 44100

            duration = len(audio) / sr
            num_chunks = int(duration // 10)

            for i in range(num_chunks):
                chunk_start = i * 10
                chunk_end = (i+1) * 10
                target = parent._process_labels(labels, chunk_start, chunk_end)

                self.data.append((
                    id,
                    chunk_start,
                    chunk_end,
                    target
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, start, end, target = self.data[idx]
        audio, sr, _ = self.parent.access(id)
        chunk_start_sample = int(start * sr)
        chunk_end_sample = int(end * sr)
        chunk = audio[chunk_start_sample:chunk_end_sample]

        if len(chunk) != 10 * sr:
            chunk = librosa.util.fix_length(chunk, size=10*sr)

        chunk_tensor = torch.FloatTensor(chunk)
        chunk_tensor = (chunk_tensor - chunk_tensor.mean()) / \
            (chunk_tensor.std() + 1e-8)

        return chunk_tensor, torch.FloatTensor(target)


class MusicNetLoader:
    def __init__(self, root, label="train"):
        self.root = Path(root)
        self.label = label
        self.data_dir = self.root / f"{label}_data"
        self.label_dir = self.root / f"{label}_labels"
        self.metadata = pd.read_csv(self.root / "musicnet_metadata.csv")
        self.ids = [int(f.stem) for f in self.data_dir.glob("*.wav")]
        self.reduced_instruments_map = {
            1: [1, 2, 3, 4, 5, 6, 7, 8],
            2: [17, 18, 19, 20, 21, 22, 23, 24],
            3: [25, 26, 27, 28, 29, 30, 31, 32],
            4: [33, 34, 35, 36, 37, 38, 39, 40],
            5: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
            6: [57, 58, 59, 60, 61, 62, 63, 64],
            7: [65, 66, 67, 68, 69, 70, 71, 72],
            8: [73, 74, 75, 76, 77, 78, 79, 80],
            9: [81, 82, 83, 84, 85, 86, 87, 88],
            10: [9, 10, 11, 12, 13, 14, 15, 16, 89, 90, 91, 92, 93, 94,
                 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
        }

        # Create reverse mapping for quick lookup
        self.reverse_instrument_map = {}
        for group, instruments in self.reduced_instruments_map.items():
            for inst in instruments:
                self.reverse_instrument_map[inst] = group

        # Verify instrument mapping with actual data
        self._validate_instrument_range()

    def _validate_instrument_range(self):
        """Check actual instrument values in labels"""
        sample_labels = pd.read_csv(self.label_dir / f"{self.ids[0]}.csv")
        instruments = sample_labels['instrument'].unique()
        print(f"Found instrument IDs: {instruments}")
        # Adjust this based on actual observations

    def keys(self):
        return self.ids

    def access(self, id):
        audio_path = self.data_dir / f"{id}.wav"
        label_path = self.label_dir / f"{id}.csv"

        audio, sr = librosa.load(audio_path, sr=None)
        labels = pd.read_csv(label_path).astype({
            'start_time': float,
            'end_time': float,
            'instrument': int
        })

        return audio, sr, labels

    def _process_labels(self, labels, chunk_start, chunk_end):
        """Map original instruments to reduced categories"""
        # Filter labels overlapping with chunk
        mask = (
            (labels['start_time'] < chunk_end) &
            (labels['end_time'] > chunk_start)
        )
        chunk_labels = labels[mask]
        instruments = chunk_labels['instrument'].unique()

        # Create multi-hot vector for 10 categories
        target = np.zeros(10, dtype=np.float32)  # Now 10 classes
        for inst in instruments:
            group = self.reverse_instrument_map.get(inst)
            if group is not None:
                # Subtract 1 for 0-based indexing
                target[group-1] = 1.0
            else:
                print(f"Warning: Instrument {inst} not in mapping")

        return target

    def get_loader(self, batch_size=500, sr=44100):
        dataset = MusicNetDataset(self, sr=sr)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def sample(self, id):
        audio, sr, labels = self.access(id)
        duration = len(audio) / sr
        num_chunks = int(duration // 10)

        samples = []
        for i in range(num_chunks):
            start = i * sr
            end = start + sr
            target = self._process_labels(labels, start, end)
            # Get group numbers (1-10) where present
            active_groups = [i+1 for i, val in enumerate(target) if val == 1]

            samples.append({
                'id': id,
                'chunk_start': start,
                'chunk_end': end,
                'instrument_groups': active_groups,
                'target': target
            })

        return pd.DataFrame(samples)
