
from caption.data import CoCoDataLoader
from caption.config import Config
from caption.model import CaptionSeq2Seq

import torch.nn as nn
import torch
from torch import optim
import datetime

config = Config()
config.train_batch_size = 16

device = torch.device(config.device)

data_loader = CoCoDataLoader(config)
data_loader.parse_json()
data_loader.construct_vocab()

criterion = nn.CrossEntropyLoss(ignore_index=data_loader.pad_id).to(device)
model = CaptionSeq2Seq(config, device, data_loader.vocab_size, criterion)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


steps = 10000

for i in range(steps):

    batch_img, batch_caption_id, batch_caption, max_len = data_loader.next_batch(mode='train',
                                                                                 normalize=True,
                                                                                 device=device)
    optimizer.zero_grad()

    loss = model(batch_img, batch_caption_id, max_len)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    if i % 20 == 0:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]. training loss: {round(loss.item(), 4)}.")
