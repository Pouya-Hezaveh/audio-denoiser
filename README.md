# Audio-Denoiser

This program gets audio files in format of .wav and denoises it using an ADALINE network written in Julia.

This filter is suitable for ultralight real-time applications.

## Start

Note that if you run the code, my function called ***ensure_packages*** would automatically checks, downloads and installs the required packages which are:
+ DSP
+ WAV
+ Plots
+ Random
+ LinearAlgebra

**The program uses 4 file path:**

1. The clean audio >> to train the network
2. The noisy audio >> to train the network
3. The input audio >> the one we need to denoise
4. A directory to save the output audio

### Example

The program is setted up on the audio files inside `./aud` as an example

This can be the voice of a pilot inside the cabine of an aircraft:

https://github.com/user-attachments/assets/0770e433-9dae-409b-a005-95d19d31c9aa

After denoising:

https://github.com/user-attachments/assets/7487c5f9-b4fa-4a35-ae5d-0888230f5529

The network effectively cancelled out annoying airborne noise


