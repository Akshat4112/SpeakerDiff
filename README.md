## SpeakerDiff: Speaker Embedding Generation using Denoising Diffusion Probabilistic Models
SpeakerDiff is a novel versatile probabilistic model that generates high-quality speech samples for Libre Speech-based speaker embeddings. We have demonstrated the effectiveness of the denoising diffusion probabilistic method in preserving the feature information in the speech while effectively anonymizing identifying information. 

### Denoising Diffusion Probabilistic Models
DDPMs define a forward diffusion process that gradually adds Gaussian noise to data x0 over T steps to get xT. This forward process can be sampled efficiently and destroys a structure, converging to a Gaussian distribution.

q(xt|xt−1) = N (xt; √1 − βtxt−1, βtI) 

q(xt|x0) = N (xt; √ ̄αtx0, (1 −  ̄αt)I) 

The reverse process is a Markov chain that learns to produce xt − 1 from xt, starting from noise xT. The reverse process is parameterized as a neural network predicting the mean and variance of the Gaussian conditionals p(xt−1|xt).

pθ(xt−1|xt) = N (xt−1; μθ(xt, t), Σθ(xt, t)) 

Training maximizes a variational lower bound on log-likelihood, linking the forward and reverse processes.


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






