#QWEN3 IMPLEMENTATION 0.6B

#Libs needed
import torch
from torch import nn
from torch.nn.functional import silu
import math
import json
from types import SimpleNamespace


config = SimpleNamespace(**json.load(open("./config.json")))


class ROPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim 
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

    def forward(self,q,k,T):
        no_pair = self.head_dim // 2

        #Pair's position
        pairT = torch.arange(no_pair)
        thetaI = 1 / (self.rope_theta ** ((2*pairT)/self.head_dim)) #(64)

        #Token postion
        tokenPos = torch.arange(T) #(6)
        mTheta = torch.outer(tokenPos,thetaI) #(m,i) (6,64)
        #m(0) is mupliplied with all values in i and m(1) same goes on , so that it could build m-rows and i-coulmns
        
        ####EXAMPLE
        #import torch
        # v1 = torch.tensor([1.0, 2.0, 3.0])
        # v2 = torch.tensor([4.0, 5.0])

        # result = torch.outer(v1, v2)
        # print(result)
        # # Output:
        # # tensor([[ 4.,  5.],
        # #         [ 8., 10.],
        # #         [12., 15.]])

        #Angle Table
        print("M_theta : ",mTheta)
        print("M theta shape : ",mTheta.shape) # (6),64)

        angleTable = torch.cat([mTheta,mTheta],dim=-1) #(6,128)

        cosTable = torch.cos(angleTable) #(6,128)
        sinTable = torch.sin(angleTable) #(6,128)



        cos = cosTable[:,None,:] #Here we are adding head slot , where it becomes [6,1,128]
        sin = sinTable[:,None,:] #Here we are adding head slot , where it becomes [6,1,128]

        def half_rot(x):
            firstHalf = x[...,:no_pair]
            secondHalf = x[...,no_pair:]

            half_rot = torch.cat([-secondHalf,firstHalf],dim=-1)
            return half_rot

        ##START FROM WHERE YOU LEFT
        
        Q_half_rotation = half_rot(q)
        K_half_rotation = half_rot(k)

        return Q_half_rotation,K_half_rotation,cos,sin



class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim 
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        #Weights
        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads,bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads,bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads,bias=False)

        self.ROT = ROPE(config)

        #Output of attn
        self.o_proj = nn.Linear(self.head_dim * self.num_attention_heads, self.hidden_size, bias=False)


    def forward(self,x):

        #PROJECTION
        Pq = self.q_proj(x)
        Pk = self.k_proj(x)
        Pv = self.v_proj(x)

        B,T,_ = x.shape #(1,6,1028)

        
        q_split = Pq.view(B,T,self.num_attention_heads,self.head_dim)
        k_split = Pk.view(B,T,self.num_key_value_heads,self.head_dim)
        v_split = Pv.view(B,T,self.num_key_value_heads,self.head_dim)

        #Here ROPE COMES IN

        q_norm = self.q_norm(q_split)
        k_norm = self.k_norm(k_split)

        q_half,k_half,cos,sin = self.ROT(q_norm,k_norm,T)

        

        q_rot = q_norm * cos + q_half * sin
        k_rot = k_norm * cos + k_half * sin

        print("Shape of q_rot : ",q_rot.shape)
        

        #Start from masking 
        maskT = torch.triu(torch.ones(T,T,dtype=bool),diagonal=1)

        #Group to split qheads to kv heads
        group = self.num_attention_heads // self.num_key_value_heads

        head_output = []
        #Per head slice
        for nthHead in range(self.num_attention_heads):
            q_h = q_rot[:,:,nthHead,:] #[1,6,nth head,128]
            k_h = k_rot[:,:,nthHead // group,:] #0/2 -> 0th head , 1/2->0.5=0th head , and goes on
            v_h = v_split[:,:,nthHead // group,:]

            scores = q_h @ k_h.transpose(-2,-1) / math.sqrt(self.head_dim)

            #Now masking the score with created mask
            scores = scores.masked_fill(maskT,float('-inf'))
            weights = torch.softmax(scores,dim=-1)

            attn_value = weights @ v_h
            head_output.append(attn_value)

        #head_output will have list of attention value list , [[..,],[..,],...]
        out = torch.cat(head_output,dim=-1)
        return self.o_proj(out)


        


class RMSNorm(nn.Module):
    def __init__(self, size,eps=1e-06):
        super().__init__()
        # self.hidden_size = config.hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self,x):
        x_type = x.dtype

        #Square and mean
        x_sm = torch.mean(x.to(torch.float32) ** 2,dim=-1 , keepdim=True)

        #Root and eps
        rms = torch.sqrt(x_sm + self.eps) #for faster efficient compute use rsqrt()
        
        #RMS NORM
        rms_norm = (x.to(torch.float32) / rms) * self.weight
        return rms_norm.to(x_type) #Return to original type


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size              # 1024
        self.intermediate_size = config.intermediate_size  # 3072
        self.initializer_range = config.initializer_range  # 0.02

        # Qwen's names → loading works later
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # weight [3072, 1024]
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # weight [3072, 1024]
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # weight [1024, 3072]

        # overwrite nn.Linear's default start with config's spread (0.02)
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=self.initializer_range)

    def forward(self, x):
        x_gate = silu(self.gate_proj(x))       # [B, T, 3072]
        x_up   = self.up_proj(x)               # [B, T, 3072]
        return self.down_proj(x_gate * x_up)   # [B, T, 1024]



class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(config=config)

        #Norms
        self.input_layernorm = RMSNorm(size=self.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size,eps=config.rms_norm_eps)

    def forward(self,x):

        x_input_norm = self.input_layernorm(x)
        x_gqa = self.self_attn(x_input_norm)
        postattn_residual = x + x_gqa
        x_postattn_norm = self.post_attention_layernorm(postattn_residual)
        x_mlp = self.mlp(x_postattn_norm)
        out = postattn_residual + x_mlp

        return out



class Qwen3Model(nn.Module):

    def __init__(self,config):
        super().__init__()

        #Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)

        #Layers
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        #Norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


    def forward(self,input_ids):

        # emb_tokens = self.embed_tokens(input_ids)

        # layers_out = self.layers(emb_tokens)
        x = self.embed_tokens(input_ids)       # [B, T, 1024]
       
        for layer in self.layers:              # layer 0, then 1, ... then 27
            x = layer(x)                       # each layer's output becomes the next one's input
       
        return self.norm(x)

        # final_norm = self.norm(layers_out)

        # return final_norm


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight     # tie them

    def forward(self, input_ids):
        hidden = self.model(input_ids)       # [B, T, 1024]
        logits = self.lm_head(hidden)        # [B, T, 151936]
        return logits
        



#COMPARING
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# hf = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen3-0.6B", torch_dtype=torch.float32, attn_implementation="eager"
# ).eval()

# mine = Qwen3ForCausalLM(config).eval()
# print(mine.load_state_dict(hf.state_dict(), strict=True))      # <All keys matched successfully>

# ids = tok("I need 500 wedding cards", return_tensors="pt").input_ids
# with torch.no_grad():
#     a = hf(ids).logits
#     b = mine(ids)

# print("max diff:", (a - b).abs().max().item())
# print("allclose:", torch.allclose(a, b, atol=1e-4))
# print("HF next word:  ", tok.decode(a[0, -1].argmax()))
# print("Mine next word:", tok.decode(b[0, -1].argmax()))       # ← your code "speaking" for the first time


#TODO: LOADING WEIGHTS
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_DIR = r"D:\models\qwen3_0_6b"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    mine = Qwen3ForCausalLM(config).eval()
    print(mine.load_state_dict(hf.state_dict(), strict=True))

    ids = tok("I am tarun", return_tensors="pt").input_ids
    with torch.no_grad():
        a = hf(ids).logits
        b = mine(ids)

    print("max diff :", (a - b).abs().max().item())
    print("allclose :", torch.allclose(a, b, atol=1e-4))
    print("HF next  :", repr(tok.decode(a[0, -1].argmax())))
    print("Mine next:", repr(tok.decode(b[0, -1].argmax())))