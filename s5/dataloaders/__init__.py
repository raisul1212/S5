# Robust imports: audio dataloader requires torchaudio which may not be
# installed / may have ABI mismatches in some conda envs.  Wrap in try/except
# so downstream lra/imdb/listops usage doesn't fail on audio-only optional dep.
try:
    from . import audio
except (ImportError, OSError) as _audio_e:
    audio = None
    import warnings as _w
    _w.warn(f"s5.dataloaders.audio not available: {_audio_e}")

from . import basic
from .base import SequenceDataset
