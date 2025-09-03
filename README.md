# Audio-Denoiser

This program gets audio files in format of .wav and denoises it using an ADALINE network written in Julia
This filter is suitable for ultralight real-time applications

## How To Start

I've developed a function called *ensure_packages* that checks, downloads and installs the required dependencies

**The program uses 4 file path:**

1. The clean audio 👉️ to train the network
2. The noisy audio 👉️ to train the network
3. The input audio 👉️ the one we need to denoise
4. A directory to save the output audio

### Example

The program is setted up on the audio files inside `./aud` as an example

This can be the voice of a pilot inside the cabine of an aircraft:

`pilot.wav`

The network effectively cancels out most airborne noise

After denoising: `./aud/output/denoised_audio.wav`
The annoying noise is gone.
