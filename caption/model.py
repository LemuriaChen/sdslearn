
import torch
import torch.nn as nn
import torchvision


# Encoder
class Encoder(nn.Module):
    """
    Encoder: convert a image to various feature maps

        from caption.data import CoCoDataLoader
        from caption.config import Config
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = Config()
        data_loader = CoCoDataLoader(config)
        data_loader.parse_json()
        data_loader.construct_vocab()
        img, _, _, _, _ = data_loader.next_batch(mode='train', normalize=True)
        enc = Encoder(config).to(device)
        out = enc(torch.Tensor(img).to(device, dtype=torch.float))
        print(out.shape)
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.enc_image_size = config.encoded_image_size

        res_net = torchvision.models.resnet101(pretrained=True)  # pre-trained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(res_net.children())[:-2]
        self.res_net = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((config.encoded_image_size, config.encoded_image_size))

        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolution blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.res_net.parameters():
            p.requires_grad = False
        # if fine-tuning, only fine-tune convolution blocks 2 through 4
        for c in list(self.res_net.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.res_net(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, config):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(config.encoder_dim, config.attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(config.decoder_dim, config.attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(config.attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, config, vocab_size):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = config.encoder_dim
        self.attention_dim = config.attention_dim
        self.embed_dim = config.embed_dim
        self.decoder_dim = config.decoder_dim
        self.vocab_size = vocab_size
        self.dropout = config.dropout

        self.attention = Attention(config)
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(config.embed_dim + config.encoder_dim, config.decoder_dim, bias=True)
        self.init_h = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.init_c = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.f_beta = nn.Linear(config.decoder_dim, config.encoder_dim)
        self.fc = nn.Linear(config.decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, max_len, device):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        predictions = torch.zeros(batch_size, vocab_size, max_len-1).to(device)
        for t in range(max_len-1):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1), (h, c))
            predictions[:, :, t] = self.fc(self.dropout(h))
        return predictions


class CaptionSeq2Seq(nn.Module):
    def __init__(self, config, device, vocab_size, criterion):
        super(CaptionSeq2Seq, self).__init__()
        self.enc = Encoder(config).to(device)
        self.attn = Attention(config).to(device)
        self.dec = DecoderWithAttention(config, vocab_size).to(device)
        self.device = device
        self.criterion = criterion

    def forward(self, batch_img, batch_caption_id, max_len):
        enc_out = self.enc(batch_img)
        predictions = self.dec(enc_out, batch_caption_id, max_len, self.device)
        loss = self.criterion(predictions, batch_caption_id[:, 1:])
        return loss
