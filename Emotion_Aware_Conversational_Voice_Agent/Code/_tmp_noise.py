"""Record room silence and report the noise floor for the Whisper silence gate."""
import argparse
import numpy as np
import pyaudio
import time

RATE = 16000
SECONDS = 5
CHUNK = 1024


def measure(device_index=None):
    pa = pyaudio.PyAudio()

    kwargs = dict(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    if device_index is not None:
        kwargs["input_device_index"] = device_index

    stream = pa.open(**kwargs)

    print(f"Recording {SECONDS}s of room silence... stay quiet.")
    time.sleep(0.5)

    frames = []
    for _ in range(int(RATE / CHUNK * SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(audio ** 2))
    dbfs = 20 * np.log10(rms + 1e-10)

    peak = np.max(np.abs(audio))
    peak_dbfs = 20 * np.log10(peak + 1e-10)

    suggested = dbfs + 6

    print(f"\nRMS:  {rms:.6f}  ({dbfs:.1f} dBFS)")
    print(f"Peak: {peak:.6f}  ({peak_dbfs:.1f} dBFS)")
    print(f"Suggested silence gate: {suggested:.1f} dBFS  (noise floor + 6 dB margin)")
    print(f"\nTo use: set asr.silence_gate_dbfs to {suggested:.1f} in gpu_config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure room noise floor.")
    parser.add_argument("--device", type=int, default=None,
                        help="PyAudio input device index (from _tmp_devices.py). "
                             "Omit to use system default.")
    args = parser.parse_args()
    measure(device_index=args.device)