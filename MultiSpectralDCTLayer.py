import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    # MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):       # (4,256,56,56)

        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        print ("weight: ", self.weight.shape)

        # plt.imshow(self.weight[1,:,:].numpy(), cmap='gray')
        # plt.show()

        x = x * self.weight     # weight:(256,56,56)  x:(4,256,56,56)
        print ("x after multip:   ", x.shape)

        result = torch.sum(x, dim=[2,3])        # result:(4,256)
        print ("result: ", result.shape)
        return result


    def build_filter(self, pos, freq, POS):     # 对应公式中i/j, h/w, H/W   一般是pos即i/j在变  # self.build_filter(t_x, u_x, tile_size_x)  self.build_filter(t_y, v_y, tile_size_y)
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)        # 为什么是乘以根号2？
    

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):          # dct_h(height), dct_w(weight), mapper_x, mapper_y, channel(256,512,1024,2048)
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)     # (256,56,56)
        c_part = channel // len(mapper_x)       # c_part = 256/16 = 16

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        print ("dct_filter: ", dct_filter.shape)                
        return dct_filter
    

if __name__ == '__main__':
    x = torch.rand([1, 16, 256, 256])
    mapper_x = [0,0,3,3,2,4,1,2,4,4,5,1,4,6,2,5]
    mapper_y = [0,0,4,6,6,3,1,4,4,5,6,5,2,2,5,1]

    model = MultiSpectralDCTLayer(256, 256, mapper_x, mapper_y, 16)
    y = model(x)
    print(y.shape)  # (4,256)




    # # Create a 4D tensor with shape (2, 3, 4, 5)
    # x = torch.ones(1, 3, 4, 4)

    # # Sum over dimensions 2 and 3
    # result = torch.sum(x, dim=[2, 3])

    # print("Original shape:", x)  # (2, 3, 4, 5)
    # print("Summed shape:", result.shape)  # (2, 3)
    # print("Result:", result)
