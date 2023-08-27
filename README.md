## SpeakerDiff: Speaker Embedding Generation using Denoising Diffusion Probabilistic Models
SpeakerDiff is a novel versatile probabilistic model that generates high-quality speech samples for Libre Speech-based speaker embeddings. We have demonstrated the effectiveness of the denoising diffusion probabilistic method in preserving the feature information in the speech while effectively anonymizing identifying information. 

### Denoising Diffusion Probabilistic Models
Denoising diffusion probabilistic models (DDPM), a promising class of generative models that gradually uses a Markov chain to convert isotropic Gaussian distribution into complex data distribution. The diffusion models serve to balance the trade-off between flexibility and traceability. We remodel the diffusion model proposed by Jonathan Ho et al. by modifying the variance scheduler and employing the entire mechanism on speaker embedding. Diffusion models operate on a noise-adding schedule without learning from the parameters to obtain salient features.

![Forward Noise](https://github.com/Akshat4112/speaker_embedding_generation_diffusion_models/blob/main/figures/Forward_noise.png)

![Backward Noise](https://github.com/Akshat4112/speaker_embedding_generation_diffusion_models/blob/main/figures/backward_noise.png)

Experiments run on 3.3Ghz AMD EPYC 7002 series. Requires Python 3.8, and these dependencies for CPU instances, Please install 'requirements.txt'

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

Linear and UNet Models are written in a model.py file, which can be modified as per the requirement

UNet model Architecture
![UNet Architecture](https://github.com/Akshat4112/speaker_embedding_generation_diffusion_models/blob/main/figures/Unet.drawio.png)
## Output Audio Samples
These audio samples are generated after passing the generated embeddings to a TTS Engine. 


Female Voice:  



Male Voice: 



## Output
Red points represent the original data points in the distribution and blue ones are generated data points. 
The t-Sne plot of Generated and Original data points. 

![Plot](https://github.com/Akshat4112/speaker_embedding_generation_diffusion_models/blob/main/figures/TSNE-Based%202D%20Plot%20of%2064%20dimensional%20Embeddings.png)

## References






