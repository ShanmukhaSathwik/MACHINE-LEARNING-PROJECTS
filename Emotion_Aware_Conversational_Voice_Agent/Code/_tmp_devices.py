"""List all audio devices so we can pick the right mic for the GPU machine."""
import pyaudio

pa = pyaudio.PyAudio()
print("All audio devices:\n")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    direction = ""
    if info["maxInputChannels"] > 0:
        direction += "IN "
    if info["maxOutputChannels"] > 0:
        direction += "OUT"
    print(
        f"  [{i:2d}] {direction:7s}  {info['name']}"
        f"  (in={info['maxInputChannels']} out={info['maxOutputChannels']}"
        f"  rate={int(info['defaultSampleRate'])})"
    )
pa.terminate()
