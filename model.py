# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 08:28:23 2018

@author: Zymieth
"""

import utils

class VibNet(nn.Module):
    '''
    Multi-head TCN for raw single channel sequential data
    Takes lists of layers to initialize an instance. path 1, path 2, path 3, shared convolutional layers, dense layers.
    e.g m = VibNet([1, 3, 1], [1, 3, 1], [1, 3, 1], [3,3,12,64,256], [256, 4])
    Attention size for the concatenated convolutional paths is hard coded and has to be determined for different choices 
    of dilation
    '''
    def __init__(self, conv_layers1, conv_layers2, conv_layers3, conv_post, layers):
        super().__init__()
        # path 1
        self.conv_layers1 = nn.ModuleList([nn.Conv1d(conv_layers1[i], conv_layers1[i + 1], kernel_size = 10, dilation=1) 
                                     for i in range(len(conv_layers1) - 1)])
        # path 2
        self.conv_layers2 = nn.ModuleList([nn.Conv1d(conv_layers2[i], conv_layers2[i + 1], kernel_size = 10, dilation=10) 
                                     for i in range(len(conv_layers2) - 1)])
        # path 3
        self.conv_layers3 = nn.ModuleList([nn.Conv1d(conv_layers3[i], conv_layers3[i + 1], kernel_size = 10, dilation=20) 
                                     for i in range(len(conv_layers3) - 1)])
        # shared convolutions
        self.conv_post = nn.ModuleList([nn.Conv1d(conv_post[i], conv_post[i + 1], kernel_size = 15, dilation=1) 
                                     for i in range(len(conv_post) - 1)])
        # dense layer
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
        # gaussian noise for data augmentation at runtime
        self.noise = utils.GaussianNoise(0.05)
        
        # several other useful modules that might or might not be used
        self.max = nn.MaxPool1d(4) 
        self.dropout = nn.Dropout(p=0.33)
        self.gp = nn.AdaptiveMaxPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(2000)
        # hardcoded attention size. has to be determined for each combination of dilations, depth of conv paths etc.
        # needs to be a multiple of 3. Optional
        self.attention = Attention(5586)
    
    def get_weights(self, layer):
        '''
        Obtain the weights of path 1, path 2, path 3 layers.
        input: layer number. returns (w1, w2, w3), the weights of each path at the specified layer
        '''
        weights1 = self.conv_layers1[layer].weight.data.cpu().numpy()
        weights2 = self.conv_layers2[layer].weight.data.cpu().numpy()
        weights3 = self.conv_layers3[layer].weight.data.cpu().numpy()
        
        return weights1, weights2, weights3
    
    def plot_latent_components(self, val, psd_flag = None):
        '''
        Takes as input a single validation time series
        Plots the latent time series activations (or power spectral density if psd_flag = True)
        for each of the path's final layer
        '''
        val = val.view(1,1,-1)
        if psd_flag:
            p1, p2, p3 = m.evaluate_paths(val)
            fig = plt.figure(figsize=(20,20), facecolor = 'white')
            fig.add_subplot(4,1,1) 
            plt.plot(get_power(val,0)[10::], color = 'black')
            fig.add_subplot(4,1,2)
            plt.plot(get_power(p1.view(1,1,-1),0)[10::], color = 'red')
            fig.add_subplot(4,1,3)
            plt.plot(get_power(p2.view(1,1,-1),0)[10::], color = 'green')
            fig.add_subplot(4,1,4)
            plt.plot(get_power(p3.view(1,1,-1),0)[10::], color = 'blue')    
        else:
            p1, p2, p3 = self.evaluate_paths(val)
            fig = plt.figure(figsize=(20,20), facecolor = 'white')
            fig.add_subplot(4,1,1) 
            plt.plot(val.view(-1).detach().cpu().numpy(), color = 'black')      
            fig.add_subplot(4,1,2)
            plt.plot(p1.view(-1).detach().cpu().numpy(), color = 'red')
            fig.add_subplot(4,1,3)
            plt.plot(p2.view(-1).detach().cpu().numpy(), color = 'green')
            fig.add_subplot(4,1,4)
            plt.plot(p3.view(-1).detach().cpu().numpy(), color = 'blue')
        
    def evaluate_paths(self, x):
        '''
        Used to obtain latent time series at inference time
        Uses the learned weights and the same architecture to manipulate validation data
        '''
        x = x.view(x.size(0), 1, -1)
        if m.training:
            x = self.noise(x)
        #save input
        s = x      
        for l in self.conv_layers1:
            l_x = l(x)
            x = torch.tanh(l_x) 
            #x = self.max(x)
        #x = self.conv_dil1(x)
        x1 = x
        x = s
        for l in self.conv_layers2:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil3(x)
        x2 = x
        x = s
        for l in self.conv_layers3:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil5(x)
        x3 = x
        return x1, x2, x3
       
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        if m.training:
            x = self.noise(x)
        #save input
        s = x
        for l in self.conv_layers1:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil1(x)
        x1 = x
        x = s
        for l in self.conv_layers2:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil3(x)
        x2 = x
        
        x = s
        for l in self.conv_layers3:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil5(x)
        x3 = x
        
        x = torch.cat((x1, x2, x3), 2)
        x = self.attention(x)
        x = x.view(s.size(0), 3, -1)
        
        for l in self.conv_post:
            l_x = l(x)
            bn = nn.BatchNorm1d(x.size())
            x = F.relu(l_x)
            #x = self.max(x)
        x = self.gp(x)
        
        x = x.view(x.size(0), -1)   
        for l in self.layers:
            l_x = l(x)
            bn = nn.BatchNorm1d(l_x.size())
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)