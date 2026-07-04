import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfGatingAttention(nn.Module):
    def __init__(self, input_dim, look_back_length):
        super(SelfGatingAttention, self).__init__()
        self.input_dim = input_dim
        self.look_back_length = look_back_length
        
        # Shared learnable matrix
        self.shared_matrix = nn.Parameter(torch.randn(look_back_length, look_back_length))
        
        # Linear layer for input-dependent residual component
        self.residual_layer = nn.Linear(input_dim, look_back_length)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, look_back_length, input_dim)
        """
        # Compute residual attention component
        residual_attention = self.residual_layer(x)  # Shape: (batch_size, look_back_length, look_back_length)
        
        # Compute final attention scores
        attention_scores = self.shared_matrix + residual_attention  # Broadcasting shared_matrix
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores
        
        # Apply attention to input
        output = torch.bmm(attention_scores, x)  # Shape: (batch_size, look_back_length, input_dim)
        return output

class ForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, look_back_length):
        super(ForecastingModel, self).__init__()
        self.sga = SelfGatingAttention(input_dim, look_back_length)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, look_back_length, input_dim)
        """
        x = self.sga(x)  # Apply Self-Gating Attention
        x = F.relu(self.fc(x))  # Apply fully connected layer
        x = x.mean(dim=1)  # Aggregate over the time dimension
        output = self.output_layer(x)  # Final output layer
        return output

if __name__ == '__main__':
    # Dummy data for testing
    batch_size = 8
    look_back_length = 10
    input_dim = 16
    hidden_dim = 32

    # Generate random input data
    x = torch.randn(batch_size, look_back_length, input_dim)

    # Initialize and test the model
    model = ForecastingModel(input_dim, hidden_dim, look_back_length)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)