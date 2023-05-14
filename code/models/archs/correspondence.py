import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
  
class Correlation_fuse_cat_corre(nn.Module):
    def __init__(self, return_x16=True, click_only=True, match_kernel=3, temperature=0.01):
        super().__init__()
        add_ch = 2 if click_only else 5
        self.click_only = click_only
        self.match_kernel = match_kernel
        self.temperature = temperature
   
        if click_only:
            self.conv_click_fea_list = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(add_ch, 64, 1, 1),
                                                     nn.MaxPool2d(2),nn.Conv2d(64, 128, 1, 1),
                                                     nn.MaxPool2d(2),nn.Conv2d(128, 256, 1, 1))
        else:
            self.conv_click_fea_list =  nn.Sequential(nn.Conv2d(add_ch, 32, 1, 1), 
                                        nn.MaxPool2d(2),nn.Conv2d(32, 64 , 1, 1),
                                        nn.MaxPool2d(2),nn.Conv2d(64, 128, 1, 1),
                                        nn.MaxPool2d(2),nn.Conv2d(128, 256, 1, 1))

        self.conv_click_cat = nn.Conv2d(256 + 256, 256 , 1, 1)
                                      
 

    def get_corre(self, theta, phi):
        # print('-------------+++++++++++------------', theta.shape, phi.shape)
    
        """
        pairwise cosine similarity
        theta: RGB feature torch.Size([4, 256, 64, 64])
        phi: seg feature
        """
       
        self.batch, self.ch, self.height, self.width = theta.shape
        if self.match_kernel == 1:
            theta = theta.view(self.batch, self.ch, -1)  # 2*256*(feature_height*feature_width)
        else:
            theta = F.unfold(theta, kernel_size=self.match_kernel, padding=int(self.match_kernel // 2))#torch.Size([4, 2304, 4096])
        
        theta = theta - theta.mean(dim=1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256 torch.Size([4, 4096, 2304])
        
        phi = F.interpolate(phi, size=(self.height, self.width), mode='bilinear')
        if self.match_kernel == 1:
            phi = phi.view(self.batch, self.ch, -1)  # 2*256*(feature_height*feature_width)
        else:
            phi = F.unfold(phi, kernel_size=self.match_kernel, padding=int(self.match_kernel // 2))
        phi = phi - phi.mean(dim=1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)#torch.Size([4, 2304, 4096])
        # print('----------------------',theta_permute.shape, phi.shape)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        f_WTA = f / self.temperature
        f_div_C = F.softmax(f_WTA.squeeze(), dim=-1)  # 2*1936*1936; torch.Size([4, 4096, 4096]) softmax along the horizontal line (dim=-1)


        return f_div_C # temperature is coefficient controling sharpness of softmax

    def propagate(self, f_div_C, ref):
        ref = F.interpolate(ref, size=(self.height, self.width), mode='bilinear')
        ref = ref.view(self.batch, self.ch, -1)
        ref = ref.permute(0, 2, 1)#torch.Size([4, 4096, 3])
        y = torch.matmul(f_div_C, ref)  # 2*1936*channel torch.Size([4, 4096, 3])
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(self.batch, self.ch, self.height, self.width)  # 2*3*44*44 torch.Size([4, 3, 64, 64])

        return y

    def forward(self, block_fea, spp_input, click_map):
        click_fea = self.conv_click_fea_list(click_map)
        corre_matrix = self.get_corre(block_fea, spp_input)
        corre_out = self.propagate(corre_matrix, click_fea)
        b,c,h,w = block_fea.shape

        corre_out = F.upsample(corre_out, size=(h,w), mode='bilinear')
        out = self.conv_click_cat(torch.cat([block_fea, corre_out], dim=1))

        return out
 
 