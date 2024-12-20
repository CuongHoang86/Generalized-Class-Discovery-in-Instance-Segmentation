import torch
import torch.nn as  nn
import torch.nn.functional as F




class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
        cha_ratio=8, spa_ratio=8, down_ratio=4):  #8
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        
        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = in_channel // cha_ratio
        
        
        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # ratio=[3,5,7]
        # ratio=[3,5,7]
        # width=[112,56,28,14]
        # width=[56,28,14,7]
        size=[18,12,8]
        k1=k2=k3=0
        if in_channel==256:
            k1=size[0]
            k2=size[1]
            k3=size[2]
            self.pool1 = nn.AdaptiveAvgPool2d(k1)
            self.pool2 = nn.AdaptiveAvgPool2d(k2)
            self.pool3 = nn.AdaptiveAvgPool2d(k3)
            
        if in_channel==512:
            k1=round(size[0]/2)
            k2=round(size[1]/2)
            k3=round(size[2]/2)
            self.pool1 = nn.AdaptiveAvgPool2d(k1)
            self.pool2 = nn.AdaptiveAvgPool2d(k2)
            self.pool3 = nn.AdaptiveAvgPool2d(k3)
           
        if in_channel==1024:
            k1=round(size[0]/4)
            k2=round(size[1]/4)
            k3=round(size[2]/4)
            self.pool1 = nn.AdaptiveAvgPool2d(k1)
            self.pool2 = nn.AdaptiveAvgPool2d(k2)
            self.pool3 = nn.AdaptiveAvgPool2d(k3)
            
        if in_channel==2048:
            k1=round(size[0]/8)
            k2=round(size[1]/8)
            k3=round(size[2]/8)
            self.pool1 = nn.AdaptiveAvgPool2d(k1)
            self.pool2 = nn.AdaptiveAvgPool2d(k2)
            self.pool3 = nn.AdaptiveAvgPool2d(k3)
        
        # print('k',k1,k2,k3)
        self.inchan=k1**2+k2**2+k3**2
        self.inter_spatial = self.inchan // spa_ratio
        
       
        
        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inchan #self.inter_spatial
            # print('hahah',num_channel_s,num_channel_s//down_ratio)
            # if num_channel_s//down_ratio ==0:
            #     down_ratio=1
            self.W_spatial = nn.Sequential(
                # nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
                #         kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(num_channel_s//down_ratio),
                # nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s, out_channels=1,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
       

        # Embedding functions for modeling relations
        # print('self.in_channel',self.in_channel,self.inter_channel)
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                                kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

            self.theta_spatial1 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                                kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

            self.theta_spatial2 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                                kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                            kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        
    def forward(self, x):
        b, c, h, w = x.size()
       
        if self.use_spatial:
        # spatial attention
            a= self.pool1(x)
            a1= self.pool2(x)
            a2= self.pool3(x)
          
            theta_xs = self.theta_spatial(a)	
            # print('a1',theta_xs.shape)
            # exit()
            theta_xs1 = self.theta_spatial1(a1)	
            # print('a2',a2.shape)
            theta_xs2 = self.theta_spatial2(a2)	
       

            phi_xs = self.phi_spatial(x)
         
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)

           
            theta_xs1 = theta_xs1.view(b, self.inter_channel, -1)
            theta_xs1 = theta_xs1.permute(0, 2, 1)


            theta_xs2 = theta_xs2.view(b, self.inter_channel, -1)
            theta_xs2 = theta_xs2.permute(0, 2, 1)


       
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
        
            theta_xs=torch.cat([theta_xs,theta_xs1,theta_xs2],dim=1)
       
            Gs = torch.matmul(theta_xs, phi_xs)
            e1,e2,e3=Gs.shape
           
            Gs_out = Gs.view(b, e2, h, w)
        
            Gs_joint = Gs_out

            
            g_xs = self.gx_spatial(x)  ###### from original

        

            g_xs = torch.mean(g_xs, dim=1, keepdim=True)

            # print('g_xs',g_xs.shape,Gs_joint.shape)
            ys = torch.cat((g_xs, Gs_joint), 1)

            # print('ys',ys.shape)

            W_ys = self.W_spatial(ys)

            
            haha=torch.sigmoid(W_ys).squeeze(1)
            # sys.exit()

            out =  torch.sigmoid(W_ys.expand_as(x))* x
            return out,haha




class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
    #   print(x.shape)
    #   print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


from model import Head
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3,feat_dim=1,mlp_out_dim=1,num_mlp_layers=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.atten1=RGA_Module(256,2240,True,False)
        self.atten2=RGA_Module(512,560,True,False)
        self.atten3=RGA_Module(1024,140,True,False)
        self.atten4=RGA_Module(2048,38,True,False)

        # self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        self.projector = Head(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    def forward(self, x):
        out = self.relu(self.batch_norm1(self.conv1(x)))
        # out = self.max_pool(out)

        out = self.layer1(out)
        # print('out123',out.shape)
        out,haha1=self.atten1(out)

        # print('out69',out.shape)
        
        
        out = self.layer2(out)
        # print('outx',out.shape)
        out,haha2=self.atten2(out)

        # exit()
        out = self.layer3(out)
        # print('outx',out.shape)
        # print('out3',out.shape)
        out,haha3=self.atten3(out)
      
        out = self.layer4(out)
        # print('outx',out.shape)
      
        out,haha4=self.atten4(out)

        # print('haha',haha.shape)
        # sys.exit()
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        haha=[haha1,haha2,haha3,haha4]

    
        out1, out2 =self.projector(out)
     
        return out1, out2,haha
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
# def ResNet18(num_classes, channels=3):
#     return ResNet(Bottleneck, [2,2,2,2], num_classes, channels)


def ResNet50(num_classes, channels=3,feat_dim=1,mlp_out_dim=1,num_mlp_layers=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels,feat_dim,mlp_out_dim,num_mlp_layers)

# def ResNet50(num_classes, channels=3,feat_dim=1,mlp_out_dim=1,num_mlp_layers=1):
    # return ResNet(Bottleneck, [2,2,2,2], num_classes, channels,feat_dim,mlp_out_dim,num_mlp_layers)
    
# def ResNet101(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

# def ResNet152(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)
