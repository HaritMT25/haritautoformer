mimport torch
import torch.nn as nn
import math

def compared_version(ver1, ver2):
    """
    Compare two version strings.
    Returns -1 if ver1 < ver2, 1 if ver1 > ver2, and True if equal.
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            continue
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

###############################################################################
# Channel Positional Encoding
###############################################################################

class ChannelPositionalEmbedding(nn.Module):
    """
    Generates a sinusoidal channel positional encoding.
    It creates a (c_in, m+1) matrix (with m fixed to 8), flattens it to (1, c_in*(m+1)),
    and repeats it n times to yield a (n, c_in*(m+1)) tensor.
    """
    def __init__(self, c_in, m=24):
        super(ChannelPositionalEmbedding, self).__init__()
        self.c_in = c_in
        self.m = int(m)  # m is now fixed to 8 by default
        pe = torch.zeros(c_in, self.m + 1).float()
        pe.requires_grad = False  # fixed encoding
        position = torch.arange(0, c_in).float().unsqueeze(1)  # shape: (c_in, 1)
        div_term = torch.exp(torch.arange(0, self.m + 1, 2).float() * -(math.log(10000.0) / (self.m + 1)))
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].size(1)])
        if (self.m + 1) > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe)

    def forward(self, n):
        # Flatten to shape (1, c_in*(m+1)) then repeat n times.
        flat = self.pe.flatten().unsqueeze(0)
        return flat.repeat(n, 1)

###############################################################################
# Other Embedding Classes
###############################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        :param c_in: Number of input channels.
        :param d_model: Model dimension.
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Set m locally to 8.
        self.m = 8  
        # Linear projection to combine token and channel encodings.
        self.concat_proj = nn.Linear(d_model + c_in * (self.m + 1), d_model)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, channel_encoding=None):
        # x: shape (batch, seq_len, c_in)
        x_emb = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # shape: (batch, seq_len, d_model)
        if channel_encoding is not None:
            # Ensure channel_encoding has shape (batch, seq_len, c_in*(m+1))
            if channel_encoding.dim() == 2:
                channel_encoding = channel_encoding.unsqueeze(0).expand(x_emb.size(0), -1, -1)
            # Concatenate token embedding and channel encoding, then project.
            x_emb = torch.cat([x_emb, channel_encoding], dim=-1)
            x_emb = self.concat_proj(x_emb)
        return x_emb

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

###############################################################################
# Modified Data Embedding with Channel Positional Encoding
###############################################################################

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        :param c_in: Number of input channels.
        :param d_model: Model dimension.
        """
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: (batch, seq_len, c_in)
        batch_size, seq_len, c_in = x.shape
        # Use m=8 locally for channel positional encoding.
        channel_encoder = ChannelPositionalEmbedding(c_in, m=24).to(x.device)
        channel_encoding = channel_encoder(seq_len)
        # Sum the token, positional, and temporal embeddings.
        x = self.value_embedding(x, channel_encoding=channel_encoding) \
            + self.position_embedding(x) \
            + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        batch_size, seq_len, c_in = x.shape
        channel_encoder = ChannelPositionalEmbedding(c_in, m=24).to(x.device)
        channel_encoding = channel_encoder(seq_len)
        x = self.value_embedding(x, channel_encoding=channel_encoding) \
            + self.temporal_embedding(x_mark)
        return self.dropout(x)
