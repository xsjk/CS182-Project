import torch
import norms
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from asteroid_filterbanks import make_enc_dec
from typing import Optional


class BaseModel(nn.Module):
    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    @property
    def sample_rate(self):
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        self.__sample_rate = new_sample_rate

    def separate(self, wav):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        assert (
            wav.shape[-2] == self.in_channels
        ), f"audio with {wav.shape[-2]} channels are not supported"
        input_device = wav.device
        model_device = next(self.parameters()).device
        wav = wav.to(model_device)
        out_wavs = self(wav)
        out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())
        out_wavs = out_wavs.to(input_device)
        return out_wavs


class BaseEncoderMaskerDecoder(BaseModel):
    def __init__(self, encoder, masker: nn.Module, decoder):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.enc_activation = nn.Identity()

    def forward(self, wav):
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = torch.tensor(wav.shape)
        # Reshape to (batch, n_mix, time)
        if wav.ndim == 1:
            wav = wav.reshape(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(1)

        # Real forward
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        decoded = self.decoder(masked_tf_rep)
        reconstructed = nn.functional.pad(
            decoded, [0, wav.shape[-1] - decoded.shape[-1]]
        )
        if len(shape) == 1:
            return reconstructed.squeeze(0)
        return reconstructed


##############################################################################################################


class SingleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(SingleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, inp):
        self.rnn.flatten_parameters()
        return self.rnn(inp)[0]


class DPRNNBlock(nn.Module):
    def __init__(self, in_chan, hid_size, num_layers=1, dropout=0):
        super(DPRNNBlock, self).__init__()
        self.intra_RNN = SingleRNN(in_chan, hid_size, num_layers, dropout=dropout)
        self.inter_RNN = SingleRNN(in_chan, hid_size, num_layers, dropout=dropout)
        self.intra_linear = nn.Linear(hid_size * 2, in_chan)
        self.intra_norm = norms.gLN(in_chan)
        self.inter_linear = nn.Linear(hid_size * 2, in_chan)
        self.inter_norm = norms.gLN(in_chan)

    def forward(self, x):
        """[batch, feats, chunk_size, num_chunks]"""
        B, N, K, L = x.size()
        output = x  # for skip connection
        # Intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # Inter-chunk processing
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        rnn_type=nn.LSTM,
        num_layers=1,
        dropout=0,
    ):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        layer_norm = norms.gLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        self.net = nn.Sequential(
            *[
                DPRNNBlock(bn_chan, hid_size, num_layers, dropout)
                for _ in range(self.n_repeats)
            ]
        )
        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)

        # Get activation function.
        self.output_act = nn.Sigmoid()

    def forward(self, mixture_w):
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = nn.functional.unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        # Map to sources with kind of 2D masks
        output = self.first_out(output)
        output = output.reshape(
            batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks
        )
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = nn.functional.fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask


class DPRNNTasNet(BaseEncoderMaskerDecoder):
    def __init__(self, n_src, sample_rate=8000):
        encoder, decoder = make_enc_dec(
            "free", kernel_size=16, n_filters=64, stride=8, sample_rate=sample_rate
        )
        super().__init__(encoder, DPRNN(encoder.n_feats_out, n_src, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, num_layers=1, dropout=0), decoder,)


##################################################################################################################################


class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, skip_chan, kernel_size, padding, dilation):
        super(Conv1DBlock, self).__init__()
        self.skip_chan = skip_chan
        self.shared_block = nn.Sequential(
            nn.Conv1d(in_chan, hid_chan, 1),
            nn.PReLU(),
            norms.gLN(hid_chan),
            nn.Conv1d(
                hid_chan,
                hid_chan,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hid_chan,
            ),
            nn.PReLU(),
            norms.gLN(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_chan, 1)

    def forward(self, x):
        """[batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_chan:
            return res_out
        return res_out, self.skip_conv(shared_out)


class TDConvNet(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
    ):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size

        layer_norm = norms.gLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList(
            Conv1DBlock(
                bn_chan,
                hid_chan,
                skip_chan,
                conv_kernel_size,
                padding=(conv_kernel_size - 1) * 2**x // 2,
                dilation=2**x,
            )
            for x in range(n_blocks)
            for _ in range(n_repeats)
        )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        self.output_act = nn.Sigmoid()

    def forward(self, mixture_w):
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask


class ConvTasNet(BaseEncoderMaskerDecoder):
    def __init__(self, n_src, out_chan=None):
        encoder, decoder = make_enc_dec(
            "free", kernel_size=16, n_filters=512, stride=8, sample_rate=8000
        )
        super().__init__(encoder, TDConvNet(
            encoder.n_feats_out,
            n_src,
            out_chan=out_chan,
            n_blocks=8,
            n_repeats=3,
            bn_chan=128,
            hid_chan=512,
            skip_chan=128,
            conv_kernel_size=3,
        ), decoder)


#############################################################################################


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # gcd=Greatest Common Divisor
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.clone().detach().long().to(signal.device)
    # frame = signal.new_tensor(frame).clone().long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class MulCatBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=False):
        super(MulCatBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)

        self.gate_rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gate_rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)

        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input):
        output = input
        # run rnn module
        rnn_output, _ = self.rnn(output)
        rnn_output = (
            self.rnn_proj(rnn_output.contiguous().view(-1, rnn_output.shape[2]))
            .view(output.shape)
            .contiguous()
        )
        # run gate rnn module
        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = (
            self.gate_rnn_proj(
                gate_rnn_output.contiguous().view(-1, gate_rnn_output.shape[2])
            )
            .view(output.shape)
            .contiguous()
        )
        # apply gated rnn
        gated_output = torch.mul(rnn_output, gate_rnn_output)
        gated_output = torch.cat([gated_output, output], 2)
        gated_output = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])
        ).view(output.shape)
        return gated_output


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


class DPMulCat(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_spk,
        dropout=0,
        num_layers=1,
        bidirectional=True,
        input_normalize=False,
    ):
        super(DPMulCat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        # create the dual path pipeline
        for i in range(num_layers):
            self.rows_grnn.append(
                MulCatBlock(
                    input_size, hidden_size, dropout, bidirectional=bidirectional
                )
            )
            self.cols_grnn.append(
                MulCatBlock(
                    input_size, hidden_size, dropout, bidirectional=bidirectional
                )
            )
            if self.in_norm:
                self.rows_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1)
        )

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = (
                output.permute(0, 3, 2, 1).contiguous().view(batch_size * d2, d1, -1)
            )
            row_output = self.rows_grnn[i](row_input)
            row_output = (
                row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
            )
            row_output = self.rows_normalization[i](row_output)
            # apply a skip connection
            if self.training:
                output = output + row_output
            else:
                output += row_output

            col_input = (
                output.permute(0, 2, 3, 1).contiguous().view(batch_size * d1, d2, -1)
            )
            col_output = self.cols_grnn[i](col_input)
            col_output = (
                col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
            )
            col_output = self.cols_normalization[i](col_output).contiguous()
            # apply a skip connection
            if self.training:
                output = output + col_output
            else:
                output += col_output

            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)
        return output_all


class Separator(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        input_normalize=False,
        bidirectional=True,
    ):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            input_normalize=input_normalize,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(
            input.type()
        )
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def create_chuncks(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )
        return segments.contiguous(), rest

    def merge_chuncks(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()  # B, N, T

    # ================= END ================= #

    def forward(self, input):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        # separate
        output_all = self.rnn_model(enc_segments)

        # merge back audio files
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


class SWave(nn.Module):
    def __init__(self, N, L, H, R, C, sr, segment, input_normalize):
        super(SWave, self).__init__()
        # hyper-parameter
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = N, L, H, R, C, sr, segment
        
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        # similar to dprnn paper, setting chancksize to sqrt(2*L)
        self.segment_size = int(np.sqrt(2 * self.sr * self.segment / (self.L / 2)))

        # model sub-networks
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(
            self.filter_dim + self.N,
            self.N,
            self.H,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            self.input_normalize,
        )
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        '''
            [batch_size, , ]
        '''
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)

        # fix time dimension, might change due to convolution operations
        T_mix = mixture.size(-1)
        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(
                mixture.shape[0], self.C, self.N, mixture_w.shape[2]
            )
            output_ii = self.decoder(output_ii)

            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            outputs.append(output_ii)
        return torch.stack(outputs)


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        # setting 50% overlap
        self.conv = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv(mixture))
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.L))(est_source)
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source

class SWaveNet(BaseModel):
    def __init__(self):
        super().__init__(sample_rate = 0)
        self.masker = SWave(N=128, L=8, H=128, R=6, C=2, sr=8000, segment=4, input_normalize=False)

    def separate(self, wav):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        assert (
            wav.shape[-2] == self.in_channels
        ), f"audio with {wav.shape[-2]} channels are not supported"
        input_device = wav.device
        model_device = next(self.parameters()).device
        wav = wav.to(model_device)
        out_wavs = self(wav)
        out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())
        out_wavs = out_wavs.to(input_device)
        return out_wavs
    

    def forward(self, wav):
        return self.masker(wav).squeeze(0)
    