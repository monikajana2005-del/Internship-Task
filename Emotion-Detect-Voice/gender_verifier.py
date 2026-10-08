import librosa
import numpy as np

def is_female_voice(file_path):
    """
    Analyzes average pitch (F0) using librosa pYIN algorithm.
    Returns True if pitch corresponds to female vocal frequency range (>= 160 Hz).
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=4.0)
        # Extract Pitch (F0) using pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        # Filter out unvoiced frames (NaNs)
        voiced_f0 = f0[~np.isnan(f0)]
        
        if len(voiced_f0) == 0:
            return False, 0.0
            
        mean_f0 = np.mean(voiced_f0)
        
        # Threshold: Average pitch >= 160 Hz indicates female voice
        is_female = mean_f0 >= 160.0
        return is_female, mean_f0
    except Exception as e:
        print(f"Gender verification error: {e}")
        return False, 0.0
    