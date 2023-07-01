from distutils.command.config import config
from unicodedata import name
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataNormalize import dataNormalize
from model import ConditionalModel
from ema import EMA
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim
from utils import * 
import wandb
import torch
import numpy as np

wandb.init(project="DDM-Project")
WANDB_API_KEY = 'ffe5b918b921d391434d044c9bc030bdef3d48de'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_PATH = '../dataset/embedding_vectors_as_list.pt'
# DATA_PATH = '../dataset/704dim_embeds.pt'
DATA_PATH = '../dataset/expected_128.pt'

name = '128_dimension_embedding_datapoints'
list_of_embeddings =torch.load(DATA_PATH, map_location='cpu')
print("Original Data has datapoints: ",len(list_of_embeddings))

settings = { 
    "datapoints": 5000,
    "num_steps": 100,
    "batch_size": 64,
    "input_dimension": 128
    }

wandb.config.datapoints = settings["datapoints"]
wandb.config.num_steps = settings["num_steps"]
wandb.config.batch_size = settings["batch_size"]
wandb.config.dimensions = settings["input_dimension"]

embeddings = list_of_embeddings[:settings['datapoints']]
b = torch.stack(embeddings)
c = b.detach().numpy()
c = dataNormalize(c)

before_name = "../figures/"+"before" + str(name)
c_PCA = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0, perplexity=20).fit_transform(c)  
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(c_PCA[:,0], c_PCA[:,1], alpha=.5, color="r")
before_title = 'Scatter plot using t-SNE' + before_name
plt.title(before_title)
plt.savefig(before_name)

data = [[x, y] for (x, y) in zip(c_PCA[:,0], c_PCA[:,1])]
table = wandb.Table(data=data, columns = ["Dimension_1", "Dimension_2"])
wandb.log({"my_custom_id_1_Before" : wandb.plot.scatter(table,"Dimension_1", "Dimension_2")})

print("The shape of input is: ", c.shape)
torch.set_default_dtype(torch.float64)
dataset = torch.tensor(c)

betas = torch.tensor([1.7e-5] * settings["num_steps"])
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=settings["num_steps"], start=1e-5, end=0.5e-2)

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def q_x(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

def q_posterior_mean_variance(x_0, x_t, t):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var

model = ConditionalModel(settings["num_steps"])
optimizer = optim.Adam(model.parameters(), lr=1e-4)

ema = EMA(0.9)
ema.register(model)

for t in range(1000):
    permutation = torch.randperm(dataset.size()[0])
    for i in range(0, dataset.size()[0], settings["batch_size"]):
        indices = permutation[i:i+settings["batch_size"]]
        batch_x = dataset[indices]
        loss = noise_estimation_loss(model, batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,settings["num_steps"])
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
    if (t % 100 == 0):
        print(loss)
        wandb.log({"Loss": loss})
        x_seq = p_sample_loop(model, dataset.shape,settings["num_steps"],alphas,betas,one_minus_alphas_bar_sqrt)
        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach()
            axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5)
            axs[i-1].set_axis_off(); 
            axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')
    
embedding_name = '../GeneratedEmbeddings/'+ str(name )
torch.save(x_seq, embedding_name)
d = cur_x.detach().numpy()
d = dataNormalize(d)

after_name = "../figures/"+"after" + str(name)

d_PCA = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0, perplexity=20).fit_transform(d)  
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(d_PCA[:,0], d_PCA[:,1], alpha=.5, color="b")
after_title = 'Scatter plot using t-SNE' + after_name
plt.title(after_title)
plt.savefig(after_name)

data = [[x, y] for (x, y) in zip(d_PCA[:,0], d_PCA[:,1])]
table = wandb.Table(data=data, columns = ["Dimension_1", "Dimension_2"])
wandb.log({"my_custom_id_1_After" : wandb.plot.scatter(table,"Dimension_1", "Dimension_2")})

fig, ax = plt.subplots(figsize=(10,8))

ax.scatter(c_PCA[:,0], c_PCA[:,1], alpha=.5, color='red')
ax.scatter(d_PCA[:,0], d_PCA[:,1], alpha=.5, color='blue')
plt_title = 'Scatter plot using t-SNE for ' + str(name)
plt.title(plt_title)
both_name = '../figures/'+'After DDM Both ' + str(name) + '.png'
plt.savefig(both_name)

#Computing Mean Across Dimensions
c = np.array(c)
d = np.array(d)
c_mean = np.mean(c, axis=1)
d_mean = np.mean(d, axis=1)
print("Original Mean is: ", c_mean)
print("Generated Mean is: ", d_mean)
diff_mean = c_mean - d_mean
mean_abs_original = np.sum(c_mean)
mean_abs_generated = np.sum(d_mean)

print("Difference between Two Means is: ", diff_mean)
wandb.log({"Original_Mean": c_mean})
wandb.log({"Generated_Mean": d_mean})
wandb.log({"Difference_Mean": diff_mean})

#Computing Variance Across Dimensions
c_var = np.var(c, axis=1)
d_var = np.var(d, axis=1)
print("Original VAR is: ", c_var)
print("Generated VAR is: ", d_var)

diff_var = c_var - d_var
var_abs_original = np.sum(c_var)
var_abs_generated = np.sum(d_var)

print("Difference between Two VAR is: ", diff_var)
wandb.log({"Original_VAR": c_var})
wandb.log({"Generated_VAR": d_var})
wandb.log({"Difference_VAR": diff_var})

print("Original AM is: ", mean_abs_original)
print("Generated AM is: ", mean_abs_generated)
print("Absolute diff in Mean is: ", (mean_abs_original-mean_abs_generated))

print("Original VAR is: ", var_abs_original)
print("Generated VAR is: ", var_abs_generated)
print("Absolute diff in VAR is: ", (var_abs_original-var_abs_generated))

cosine = cosine_similarity(c, d)
print("Cosine Similarity is: ", cosine)
print("Cosine Mean is: ", np.mean(cosine))



