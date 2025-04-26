import parselmouth
import numpy as np
import librosa

def extract_acoustic_readings(audio_path):
    """
    Extracts acoustic features from an audio file.
    """
    try:
        sound = parselmouth.Sound(audio_path)
        y, sr = librosa.load(audio_path, sr=None)

        # Pitch → f0
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        times = pitch.xs()
        total = len(f0)
        voiced = np.where(f0 > 0)[0]

        mean_f0 = np.mean(f0[voiced]) if voiced.size else np.nan
        voice_period = 1.0 / mean_f0 if mean_f0 > 0 else np.nan
        voiced_ratio = voiced.size / total * 100 if voiced.size else np.nan

        # Jitter
        if voiced.size >= 2:
            periods = 1.0 / f0[voiced]
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100
        else:
            jitter = np.nan

        # Shimmer
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_t = np.arange(len(rms)) * 512 / sr
        if voiced.size >= 2:
            amps = [rms[(np.abs(rms_t - t)).argmin()] for t in times[voiced]]
            shimmer = np.mean(np.abs(np.diff(amps))) / np.mean(amps) * 100
        else:
            shimmer = np.nan

        # HNR
        try:
            harm = sound.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harm, "Get mean", 0, sound.duration)
        except:
            hnr = np.nan

        # Formants
        try:
            form = sound.to_formant_burg()
            mid = sound.duration / 2
            f1 = form.get_value_at_time(1, mid)
            f2 = form.get_value_at_time(2, mid)
            f3 = form.get_value_at_time(3, mid)
        except:
            f1 = f2 = f3 = np.nan

        return {
            "jitter": jitter,
            "shimmer": shimmer,
            "mean_f0": mean_f0,
            "hnr": hnr,
            "voice_period": voice_period,
            "voiced_ratio": voiced_ratio,
            "formants": {"F1": f1, "F2": f2, "F3": f3}
        }

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")
