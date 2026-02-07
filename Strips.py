def horizontal_stripes_midpoints(batch: torch.Tensor) -> torch.Tensor:
    ### START CODE HERE ###
    # Pad the input horizontally with 0s to detect stripes at edges
    # Shape becomes (B, C, H, W+2)
    padded = torch.nn.functional.pad(batch, (1, 1), mode='constant', value=0)
    
    # Calculate differences between adjacent pixels along the width dimension
    # Shape becomes (B, C, H, W+1)
    diff = padded[..., 1:] - padded[..., :-1]
    
    # Find indices of stripe starts (diff == 1) and ends (diff == -1)
    # .nonzero() returns a tensor of shape (N, 4) -> [batch_idx, channel_idx, row_idx, col_idx]
    starts = (diff == 1).nonzero()
    ends = (diff == -1).nonzero()
    
    # Calculate the column index for midpoints
    # Note: 'starts' col index is the actual start. 'ends' col index is (actual end + 1).
    # Formula: floor((w_start + w_end) / 2)
    # Substituion: floor((start_col + (end_col - 1)) / 2)
    mid_cols = (starts[:, 3] + ends[:, 3] - 1) // 2
    
    # Create the output tensor
    out = torch.zeros_like(batch)
    
    # Set the calculated midpoints to 1.0 using advanced indexing
    # We reuse the batch, channel, and row indices from 'starts'
    out[starts[:, 0], starts[:, 1], starts[:, 2], mid_cols] = 1.0
    
    return out
    ### END CODE HERE ###

def horizontal_stripes_midpoints(batch: torch.Tensor) -> torch.Tensor:
    ### START CODE HERE ###
    # Pad the width dimension (last dim) with 0s on left and right
    # Input shape: (B, C, H, W) -> Padded shape: (B, C, H, W+2)
    padded = torch.nn.functional.pad(batch, (1, 1), mode='constant', value=0.0)
    
    # Calculate discrete difference along the width
    # Shape becomes (B, C, H, W+1)
    diff = padded[..., 1:] - padded[..., :-1]
    
    # Find indices where stripes start (0 -> 1) and end (1 -> 0)
    # nonzero() returns coordinates in (N, 4) shape: [b, c, h, w]
    starts = (diff == 1.0).nonzero()
    ends = (diff == -1.0).nonzero()
    
    # Extract the width coordinates (last column of the indices)
    w_start = starts[:, -1]
    w_end_marker = ends[:, -1]
    
    # The end marker from diff is the index of the 0 *after* the stripe.
    # The actual last pixel of the stripe is w_end_marker - 1.
    w_end = w_end_marker - 1
    
    # Calculate midpoint: floor((start + end) / 2)
    w_mid = (w_start + w_end) // 2
    
    # Create output tensor
    output = torch.zeros_like(batch)
    
    # Use advanced indexing to set midpoints to 1.
    # We use the b, c, h coordinates from the 'starts' tensor 
    # (which are identical to 'ends' for the corresponding stripe).
    output[starts[:, 0], starts[:, 1], starts[:, 2], w_mid] = 1.0
    
    return output
    ### END CODE HERE ###




