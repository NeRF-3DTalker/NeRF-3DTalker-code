import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lib.quantizer import VectorQuantizer
from models.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from base import BaseModel


                
class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """

    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args, args.in_dim)
        self.quantize = VectorQuantizer(args.n_embed,
                                        args.zquant_dim,
                                        beta=0.25)
        self.quantize_au = VectorQuantizer(500,
                                        args.zquant_dim,
                                        beta=0.25)
        self.args = args
   


    def encode(self, x, x_a=None):
        h = self.encoder(x) ## x --> z'
        h = h.view(x.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim)
        h = h.view(x.shape[0], -1, self.args.zquant_dim)
        #print(h.shape) 1 16384 64
        #self.quantize.eval()
        #with torch.no_grad():
        quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        #print(quant.shape) 1 16384 64
        return quant, emb_loss, info
    
    def encode_au(self, x, x_a=None):
        h = x.view(x.shape[0], -1, self.args.zquant_dim)
        #print(h.shape) 1 16384 64
        #self.quantize_au.eval()
        #with torch.no_grad():
        quant, emb_loss, info = self.quantize_au(h) ## finds nearest quantization
        #print(quant.shape) 1 16384 64
        return quant, emb_loss, info

    def decode(self, quant):
        #BCL
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.args.face_quan_num+200, self.args.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1,  512).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = self.decoder(quant) ## z' --> x

        return dec

    def forward(self, x, AU_fea):
        #template = template.unsqueeze(1) #B,V*3 -> B, 1, V*3
        x = x #- template

        ###x.shape: [B, L C]
        quant, emb_loss, info = self.encode(x)
        ### quant [B, C, L]
        #print(quant.shape)# 1 64 16384
        quant_au, emb_loss_au, info_au = self.encode_au(AU_fea)

        quant=torch.cat([quant, quant_au], dim=2)
        #print(quant.shape)
        dec = self.decode(quant)

        dec = dec# + template
        #print(dec.shape)
        return dec, emb_loss+emb_loss_au, info


    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0
        ##########
        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(512)
        #self.max_pool = nn.AdaptiveMaxPool2d(512)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #avg_out = self.avg_pool(x)
        #max_out = self.max_pool(x)
        #avg_out = self.fc(
        max_out = self.fc(x)
        #print('avg_out:',avg_out.shape)        
        out = max_out
        #print(out.shape)        
        return out
class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args):
    super().__init__()
    self.args = args
    size = self.args.in_dim
    dim = self.args.hidden_size
    #self.conv1 = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.ReLU(inplace=True))
    
    self.conv1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    #nn.InstanceNorm2d(64, affine=True),    
    nn.ReLU(inplace=True),
    #ChannelAttention(64),    
    nn.Conv2d(64, 1, kernel_size=3, padding=1),
    #nn.InstanceNorm2d(64, affine=True),    
    #nn.ReLU(inplace=True),
    #ChannelAttention(64), 
    #nn.Conv2d(64, 1, kernel_size=3, padding=1)   
    #nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
    #nn.InstanceNorm2d(1, affine=True)    
    )
    '''
    #F.interpolate(input_image, scale_factor=2, mode='bilinear', align_corners=False) 
    self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True))
    self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True))
    self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True))
    '''
    #self.vertice_mapping1 = nn.Sequential(nn.Linear(131072,32768), nn.LeakyReLU(self.args.neg, True))
    #self.vertice_mapping2 = nn.Sequential(nn.Linear(32768,8192), nn.LeakyReLU(self.args.neg, True))
    self.vertice_mapping3 = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(self.args.neg, True))
    
    if args.quant_factor == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
        for _ in range(1, args.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.args.neg, True),
                        nn.InstanceNorm1d(dim, affine=args.INaffine),
                        nn.MaxPool1d(2)
                        )]
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
                self.args.num_hidden_layers,
        num_attention_heads=\
                self.args.num_attention_heads,
        intermediate_size=\
                self.args.intermediate_size)
    self.encoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.encoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs=inputs.squeeze(0)
    inputs=self.conv1(inputs)
    #inputs=inputs.contiguous().view(inputs.shape[0],64, -1)
    #inputs=self.conv1(inputs)
    #print(inputs.shape)
    '''
    inputs=self.conv1(inputs)
    inputs=self.conv2(inputs)
    inputs=self.conv3(inputs)
    inputs=self.conv4(inputs)
    #print(inputs.shape)    
    inputs=inputs.contiguous().view(inputs.shape[0],64, -1)
    '''
    #print(inputs.shape)
    #inputs = self.vertice_mapping1(inputs)
    #inputs = self.vertice_mapping2(inputs)
    #print(inputs.shape)
    #inputs=inputs.unsqueeze(0)
    inputs = self.vertice_mapping3(inputs)
        
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args, out_dim, is_audio=False):
    super().__init__()
    self.args = args
    size=self.args.hidden_size
    dim=self.args.hidden_size
    self.fc = nn.Linear(612, 512)
    self.expander = nn.ModuleList()
    if args.quant_factor == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))                      
        num_layers = args.quant_factor+2 \
            if is_audio else args.quant_factor

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.args.neg, True),
                                nn.InstanceNorm1d(dim, affine=args.INaffine),
                                ))
    self.decoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
            self.args.num_hidden_layers,
        num_attention_heads=\
            self.args.num_attention_heads,
        intermediate_size=\
            self.args.intermediate_size)
    self.decoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.decoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

    self.vertice_map_reverse = nn.Linear(args.hidden_size,out_dim)
    '''
    #self.fc = nn.Linear(64, 128*32*32)
    self.dconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
    self.dconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
    self.dconv3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
    self.dconv4 = nn.Sequential(nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1))
    #self.dconv5 = nn.Sequential(nn.Conv2d(48, 16, 5, stride=1, padding=2), nn.ReLU(inplace=True))
    #self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)
    '''
    self.downsample_model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3,stride=1,  padding=1),
    #nn.InstanceNorm2d(64, affine=True),    
    nn.ReLU(inplace=True),
    #ChannelAttention(64),    
    nn.Conv2d(64, 3, kernel_size=3,stride=1,  padding=1),
    #nn.InstanceNorm2d(64, affine=True),    
    #nn.ReLU(inplace=True),
    #ChannelAttention(64),    
    #nn.Conv2d(64, 3, kernel_size=5,stride=2,  padding=2),
    #nn.InstanceNorm2d(3, affine=True)    
    )
    

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    
    inputs=self.fc(inputs)
    #print(inputs.shape)
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)
    inputs = inputs.permute(0,2,1) #BLC
    #print(inputs.shape)
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.vertice_map_reverse(decoder_features)
    '''
    pred_recon=pred_recon.contiguous().view(inputs.shape[0],-1, 32,32)
    pred_recon=self.dconv1(pred_recon)
    pred_recon=self.dconv2(pred_recon)
    pred_recon=self.dconv3(pred_recon)
    pred_recon=self.dconv4(pred_recon)
    '''
    #print(pred_recon.shape)
    #pred_recon=pred_recon.contiguous().view(pred_recon.shape[0],512, -1)
    #print(pred_recon.shape)
    pred_recon = self.downsample_model(pred_recon)
    #print(pred_recon.shape)
    return pred_recon