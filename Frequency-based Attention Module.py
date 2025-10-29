
import torch
import torch.nn as nn
import torch.fft  # For DCT, we use Fourier Transform equivalents


########## source: COPILOT and copilot says:
#The code I provided for the Frequency-based Attention Module using DCT is an original example I created based on my understanding of the topic, 
# combining concepts from frequency-domain analysis (DCT) and attention mechanisms 
# widely discussed in neural network literature. It is not directly copied from any specific source

class FrequencyAttention(nn.Module):
    """
    Frequency-based Attention Module using DCT
    """
    def __init__(self, in_channels):
        super(FrequencyAttention, self).__init__()
        self.in_channels = in_channels

        # Learnable weights for high and low-frequency components
        self.high_freq_weight = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.low_freq_weight = nn.Parameter(torch.ones(in_channels, 1, 1))

        # A simple activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.size()

        # Apply Discrete Cosine Transform (DCT)
        dct = torch.fft.fft2(x, norm="ortho").real  # Use FFT for DCT equivalent

        # Split into low-frequency and high-frequency components
        low_freq = dct[:, :, :h//2, :w//2]  # Top-left corner for low frequencies
        high_freq = dct[:, :, h//2:, w//2:]  # Bottom-right corner for high frequencies

        # Aggregate frequency components (mean pooling here)
        low_freq_mean = low_freq.mean(dim=(2, 3), keepdim=True)
        high_freq_mean = high_freq.mean(dim=(2, 3), keepdim=True)

        # Compute attention weights
        low_weighted = low_freq_mean * self.low_freq_weight
        high_weighted = high_freq_mean * self.high_freq_weight

        # Combine attention weights
        attention = self.sigmoid(low_weighted + high_weighted)

        # Apply attention to the input feature map
        output = x * attention

        return output

# Example usage
if __name__ == "__main__":
    # Create the module
    freq_attention = FrequencyAttention(in_channels=64)

    # Example input tensor
    input_tensor = torch.rand(8, 64, 32, 32)  # (batch_size, channels, height, width)

    # Forward pass
    output = freq_attention(input_tensor)
    print("Output shape:", output.shape)
