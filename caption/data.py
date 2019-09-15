
import json
from collections import Counter
import os
from random import sample
import numpy as np
from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import wrap
import torch


class CoCoDataLoader(object):

    def __init__(self, config):

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.images = None

        self._word2id = None
        self._id2word = None

        self.pad_id = 0
        self.unk_id = 1
        self.start_id = 2
        self.end_id = 3
        self.vocab_size = 4

        self.config = config
        self.image_folder = config.image_folder

        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.img_sd = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def parse_json(self):

        # json_path = 'caption/data/coco/dataset_coco.json'
        with open(self.config.json_path, 'r') as f:
            data = json.load(f)
        assert data['dataset'] == 'coco'
        train_idx, val_idx, test_idx = [], [], []

        images = data['images']
        for idx in range(len(images)):
            if images[idx]['split'] in ['train', 'restval']:
                train_idx.append(idx)
            elif images[idx]['split'] == 'val':
                val_idx.append(idx)
            elif images[idx]['split'] == 'test':
                test_idx.append(idx)
            else:
                raise Exception

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.images = images

    def construct_vocab(self, min_word_freq=2):

        assert self.train_idx is not None
        word_freq = Counter()
        for idx in self.train_idx:
            img = self.images[idx]
            for sentences in img['sentences']:
                word_freq.update(sentences['tokens'])

        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word2id = {k: v + 4 for v, k in enumerate(words)}

        word2id['<pad>'] = self.pad_id
        word2id['<unk>'] = self.unk_id
        word2id['<start>'] = self.start_id
        word2id['<end>'] = self.end_id
        id2word = {word2id[w]: w for w in word2id}

        print(f'construct vocabulary complete. {len(id2word)} words in total.')
        print(f'the word of id <88> is <{id2word.get(88)}>.')
        print(f"the id of word <love> is <{word2id.get('love')}>.")

        self.vocab_size = len(id2word)
        self._word2id = word2id
        self._id2word = id2word

    def word_to_id(self, word):
        assert self._word2id is not None
        return self._word2id.get(word) if self._word2id.get(word) is not None else self.unk_id

    def id_to_word(self, word_id):
        assert self._id2word is not None
        assert word_id < self.vocab_size
        return self._id2word.get(word_id)

    def ids_to_sentence(self, word_id_list):
        return ' '.join([self.id_to_word(word_id) for word_id in word_id_list])

    def next_batch(self, mode, normalize=True, device=None):
        idx = None
        if mode in 'train':
            idx = sample(self.train_idx, self.config.train_batch_size)
        elif mode == 'val':
            idx = sample(self.val_idx, self.config.val_batch_size)
        else:
            pass
        assert idx is not None

        img_pixels, img_captions_id, img_captions, _, max_len = self.get_batch(idx, normalize)

        if device:
            img_pixels = torch.tensor(img_pixels).to(device, dtype=torch.float)
            img_captions_id = torch.tensor(img_captions_id).to(device, dtype=torch.long)

        return img_pixels, img_captions_id, img_captions, max_len

    def get_batch(self, idx, normalize):
        # read images and captions
        img_pixels, img_captions = [], []
        for i in idx:
            img = self.images[i]
            img_path = os.path.join(self.image_folder, img['filepath'], img['filename'])
            # img_path = 'caption/data/coco/img/val2014/COCO_val2014_000000547597.jpg'
            img_pixel = imread(img_path)
            img_pixels.append(self.process_image(img_pixel, normalize))
            all_tokens = [sentence['tokens'] for sentence in img['sentences']]
            sampled_token = sample(all_tokens, 1)[0]
            img_captions.append(sampled_token)
        img_pixels = np.array(img_pixels)
        img_captions_id = [[self.start_id] + [self.word_to_id(word) for word in sentence] + [self.end_id]
                           for sentence in img_captions]
        max_len = min(max([len(_) for _ in img_captions_id]), self.config.max_dec_len)
        img_captions_id = [_[:max_len] for _ in img_captions_id]
        img_captions_id = [_ + [self.pad_id] * (max_len - len(_)) for _ in img_captions_id]
        img_captions_id = np.array(img_captions_id)
        mask_matrix = (img_captions_id != self.pad_id) + 0
        return img_pixels, img_captions_id, img_captions, mask_matrix, max_len

    def process_image(self, img_pixel, normalize):
        if len(img_pixel.shape) == 2:
            img_pixel = img_pixel[:, :, np.newaxis]
            img_pixel = np.concatenate([img_pixel, img_pixel, img_pixel], axis=2)
        img_pixel = np.array(Image.fromarray(img_pixel).resize((256, 256)))
        img_pixel = img_pixel.transpose([2, 0, 1])
        assert img_pixel.shape == (3, 256, 256)
        assert np.max(img_pixel) <= 255
        if normalize:
            img_pixel = img_pixel / 255
            img_pixel = (img_pixel - self.img_mean) / self.img_sd
        return img_pixel

    @staticmethod
    def visualize(images, captions, rows=2, columns=2, save=False):
        fig = plt.figure(figsize=(8, 8))
        for i in range(columns * rows):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(images[i % (rows*columns)].transpose(1, 2, 0))
            plt.axis('off')
            plt.title('\n'.join(wrap(' '.join(captions[i % (rows*columns)]), 40)))
            plt.tight_layout()
        if save:
            plt.savefig('captions_visualization.jpg', format='jpg', dpi=1200)
        else:
            plt.show()


if __name__ == '__main__':

    from caption.config import Config

    data_loader = CoCoDataLoader(Config())
    data_loader.parse_json()
    data_loader.construct_vocab()

    batch_img, _, batch_caption, _ = data_loader.next_batch(mode='train', normalize=False)
    data_loader.visualize(np.array(batch_img.tolist()), batch_caption, save=False)
