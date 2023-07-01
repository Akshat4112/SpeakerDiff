## Speaker Embedding Generation using Denoising Diffusion Probalistic Models

### Denoising Diffusion Probalistic Models
![Forward Noise](https://github.tik.uni-stuttgart.de/FlorianLux/SpeakerEmbeddingGenerationDenoisingDiffusion/blob/master/figures/Forward_noise.png)

![Backward Noise](https://github.tik.uni-stuttgart.de/FlorianLux/SpeakerEmbeddingGenerationDenoisingDiffusion/blob/master/figures/backward_noise.png)

Experiments run on 3.3Ghz AMD EPYC 7002 series. Requires Python 3.8, and these dependencies for CPU instances, please install 'requirements.txt'

```bash
pip3 install -r requirements.txt
```

## Dataset

There are 3 types of embeddings generated from LibreSpeech Corpus: 
1. 64 Dimensional, which has 19k samples
2. 128 Dimensional, which has 49k samples
3. 704 Dimensional, which has 5k samples

## Training
```bash
python3 main.py
```

## Model

Linear and UNet Model are written in model.py file, which can be modified as per the requirement

UNet model Architecture
![UNet Architecture](https://github.tik.uni-stuttgart.de/FlorianLux/SpeakerEmbeddingGenerationDenoisingDiffusion/blob/master/figures/Unet.drawio.png)
## Output Audio Samples
These audio samples are generated after passing the generated embeddings to a TTS Engine. 


Female Voice:  

https://media.github.tik.uni-stuttgart.de/user/5258/files/afb624ab-1620-49a0-8119-2d4fa8310c27

Male Voice: 

https://media.github.tik.uni-stuttgart.de/user/5258/files/98ad29e5-c7ce-4e9d-a2c8-7badc3c942d5

## Output
Red points represents original data points in the distribution and blue ones are generated datapoints. 
T-Sne plot of Generated and Original datapoints. 

![Plot](https://github.tik.uni-stuttgart.de/FlorianLux/SpeakerEmbeddingGenerationDenoisingDiffusion/blob/master/figures/TSNE-Based%202D%20Plot%20of%2064%20dimensional%20Embeddings.png)

## References






