
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

# if __name__ == '__main__':
    # print(get_freq_indices('top4'))
    # # print(get_freq_indices('bot4'))
    # # print(get_freq_indices('low4'))
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import math

    N = 8
    # img = np.zeros((N, N))

    # # for u in range(N):
    # #     for v in range(N):
    # u = 0
    # v = 0
    # result = 0
    # for u in range(N):
    #     for v in range(N):
    #         for x in range(N):
    #             for y in range(N):        
    #                 c = (math.cos(math.pi * u * (x + 0.5) / N) / math.sqrt(N) ) * (math.cos(math.pi * v * (y + 0.5) / N) / math.sqrt(N))
    #                 result = result + c
    #         # if u == 0:
    #         #     result = result * math.sqrt(2)/N
    #         # else:
    #         #     result = result
    #         img[u, v] = result
    
    #         # print(result)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(img, cmap='gray')
    # ax.axis('off')
    # plt.show()


#########################################################################################
######################### 2D DCT Basis Function ##########################################
#########################################################################################
import numpy as np
import matplotlib.pyplot as plt

# def generate_dct_basis(size=8):
def dct_basis(u, v, x, y, N):
        # 2D DCT basis function
        alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
        alpha_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
        return alpha_u * alpha_v * np.cos(((2 * x + 1) * u * np.pi) / (2 * N)) *  np.cos(((2 * y + 1) * v * np.pi) / (2 * N))

N = 8
    # basis_signals = np.zeros((N, N, N, N))
basis_signals = np.zeros((N, N))
img = np.zeros((N, N))

u=7
v=0
for x in range(N):
    for y in range(N):
        basis_signals[x, y] = dct_basis(u, v, x, y, N)
    img = basis_signals


    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.imshow(img, cmap='gray')
    # ax.axis('off')
plt.show()



#     # Plot the basis signals
#     fig, axes = plt.subplots(N, N, figsize=(12, 12))
#     for u in range(N):
#         for v in range(N):
#             axes[u, v].imshow(basis_signals[u, v], cmap='gray', interpolation='nearest')
#             axes[u, v].axis('off')
#             axes[u, v].set_title(f"u={u}, v={v}", fontsize=8)

#     plt.tight_layout()
#     plt.show()

# # Generate and visualize 8x8 DCT basis signals
# generate_dct_basis(size=8)
