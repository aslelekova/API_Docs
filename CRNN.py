import collections
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

alphabet = string.ascii_lowercase + string.digits
hidden_size = 256
vocab_size = len(alphabet) + 1  # extra character for blank symbol
bidirectional = True
dropout = 0.2
weight_decay = 1e-5
momentum = 0.9
clip_norm = 5
max_epoch = 150
device = 'cpu'


class blockCNN(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, padding, stride=1):
        super(blockCNN, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = kernel_size
        self.padding = padding
        # layers
        self.conv = nn.Conv2d(in_nc, out_nc,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_nc)

    def forward(self, batch, use_bn=False, use_relu=False,
                use_maxpool=False, maxpool_kernelsize=None):
        batch = self.conv(batch)
        if use_bn:
            batch = self.bn(batch)
        if use_relu:
            batch = F.relu(batch)
        if use_maxpool:
            assert maxpool_kernelsize is not None
            batch = F.max_pool2d(batch, kernel_size=maxpool_kernelsize, stride=2)
        return batch


class blockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional, dropout=0):
        super(blockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional
        # layers
        self.gru = nn.GRU(in_size, hidden_size, bidirectional=bidirectional)

    def forward(self, batch, add_output=False):
        batch_size = batch.size(1)
        outputs, hidden = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class CRNN(nn.Module):
    def __init__(self, hidden_size: int,
                 vocab_size: int,
                 bidirectional: bool = True,
                 dropout: float = 0.5):
        super(CRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        # make layers
        # convolutions
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        # PRETRAINED^

        self.cn6 = blockCNN(256, 256, kernel_size=3, padding=1)
        # RNN + Linear
        self.linear1 = nn.Linear(1024, 256)
        self.gru1 = blockRNN(256, hidden_size, hidden_size,
                             dropout=dropout,
                             bidirectional=bidirectional)
        self.gru2 = blockRNN(hidden_size, hidden_size, vocab_size,
                             dropout=dropout,
                             bidirectional=bidirectional)
        self.linear2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, batch: torch.Tensor):
        batch_size = batch.size(0)
        # convolutions
        batch = self.resnet(batch)
        batch = self.cn6(batch, use_relu=True, use_bn=True)
        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.view(batch_size, n_channels, -1)
        batch = self.linear1(batch)
        # rnn layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)
        # output
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        return batch


class CaptchaDataset(Dataset):
    def __init__(self, img_dir: str):
        pathes = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)
        self.img_dir = img_dir
        self.pathes = [os.path.join(abspath, path) for path in pathes]
        self.list_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):
        path = self.pathes[idx]
        text = self.get_filename(path)
        if '_' in text:
            text = text.split('_')[0]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, text

    def get_filename(self, path: str) -> str:
        return os.path.basename(path).split('.')[0].lower().strip()

    def transform(self, img) -> torch.Tensor:
        return self.list_transforms(img)


class strLabelConverter(object):
    def __init__(self, alphabet: str, ignore_case: bool = True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.char2idx = {}
        for i, char in enumerate(alphabet):
            self.char2idx[char] = i + 1
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.char2idx[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
        return texts


def decode_prediction(logits: torch.Tensor,
                      label_converter: strLabelConverter) -> str:
    tokens = logits.softmax(2).argmax(2)
    tokens = tokens.squeeze(1).numpy()

    # convert tor stings tokens
    tokens = ''.join([label_converter.idx2char[token]
                      if token != 0 else '-'
                      for token in tokens])
    tokens = tokens.split('-')

    # remove duplicates
    text = [char
            for batch_token in tokens
            for idx, char in enumerate(batch_token)
            if char != batch_token[idx - 1] or len(batch_token) == 1]
    text = ''.join(text)
    return text


model = CRNN(hidden_size=hidden_size, vocab_size=vocab_size,
             bidirectional=bidirectional, dropout=dropout).to('cpu')
model.load_state_dict(torch.load("crnn.pt", map_location='cpu'))


def predict(dir_path):
    dataset = CaptchaDataset(dir_path)

    idx = np.random.randint(len(dataset))

    with torch.no_grad():
        model.eval()
        img, target_text = dataset[idx]
        img = img.unsqueeze(0)
        logits = model(img.to(device))

    label_converter = strLabelConverter(alphabet)
    pred_text = decode_prediction(logits.cpu(), label_converter)
    return pred_text


if __name__ == '__main__':
    print(predict('tmp'))
