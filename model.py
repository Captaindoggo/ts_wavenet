import torch
from torch import nn

import torchaudio

class Block(nn.Module):

    def __init__(self, n_mels, dilation, r, s=240):
        super(Block, self).__init__()
        self.conditional = nn.Conv1d(n_mels, 2 * r, kernel_size=1)
        self.dilated = nn.Conv1d(r, 2 * r, kernel_size=2, dilation=dilation)
        self.left_pad = nn.ConstantPad1d((dilation, 0), 0)

        self.t = nn.Tanh()
        self.s = nn.Sigmoid()
        self.skip = nn.Conv1d(r, s, kernel_size=1)
        self.residual = nn.Conv1d(r, r, kernel_size=1)
        self.r = r

    def forward(self, mel, wav):
        wav_dilated = self.left_pad(wav)
        mel = self.conditional(mel)
        wav_dilated = self.dilated(wav_dilated)
        xt = mel[:, :self.r] + wav_dilated[:, :self.r]  # ??
        xs = mel[:, self.r:] + wav_dilated[:, self.r:]
        xt = self.t(xt)
        xs = self.s(xs)
        x = xt * xs
        return self.skip(x), self.residual(x) + wav

class WaveNet(nn.Module):

    def __init__(self, device, hop_len=256, upsample_kernel=512, n_mels=80, caus_ks=256, r=120, s=240, a=256,
                 n_blocks=16):
        super(WaveNet, self).__init__()
        self.hop_len = hop_len
        self.a = a
        self.upsample = nn.ConvTranspose1d(n_mels, n_mels, kernel_size=upsample_kernel, stride=hop_len,
                                           padding=upsample_kernel // 2)

        self.left_pad = nn.ConstantPad1d((caus_ks - 1, 0), 0)
        self.causal = nn.Conv1d(1, r, kernel_size=caus_ks)

        dilations = [2 ** i for i in range(8)]
        dilations.extend(dilations)
        self.blocks = nn.ModuleList([Block(n_mels=n_mels, dilation=i, r=r, s=s) for i in dilations])

        self.relu1 = nn.ReLU(True)
        self.out = nn.Conv1d(s, a, kernel_size=1)
        self.relu2 = nn.ReLU(True)
        self.end = nn.Conv1d(a, a, kernel_size=1)
        self.device = device

    def forward(self, mel, wav, UP=True):
        if UP:
            mel = self.upsample(mel)
            mel = mel[:, :, 1:]

        wav = self.left_pad(wav).unsqueeze(1)
        wav = self.causal(wav)
        skips = None
        for block in self.blocks:
            skip, wav = block(mel, wav)
            if skips is None:
                skips = skip
            else:
                skips = skips + skip
        res = self.relu1(skips)
        res = self.out(res)
        res = self.relu2(res)
        return self.end(res)

    def inference(self, mel):
        mel = self.upsample(mel)
        res = self.forward(mel[:, :, 0].unsqueeze(2), torch.tensor([[0.0]]).to(self.device),
                           UP=False)  # предсказываем первый вав по первому мелу и 0.0
        out = torch.zeros(mel.shape[2]).to(self.device)  # создаем тензор длины апнутого мела
        out[0] = torch.argmax(res, dim=1).type(torch.FloatTensor)[0][
            0]  # первый эелемент уже предсказали, остальное в цикле
        pbar = tqdm_notebook(total=mel.shape[2] - 1)

        for i in range(1, mel.shape[2]):
            res = self.forward(mel[:, :, 1:i + 1], out[:i].unsqueeze(0), UP=False)
            out[i] = torch.argmax(res, dim=1).type(torch.FloatTensor)[0][-1]
            pbar.update(1)
        pbar.close()
        return out

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])