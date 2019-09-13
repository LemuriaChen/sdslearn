
from caption.data import DataLoader, visualize

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=5)
parser.add_argument('--val_batch_size', type=int, default=5)
parser.add_argument('--test_batch_size', type=int, default=5)
parser.add_argument('--max_dec_len', type=int, default=50)

parser.add_argument('--image_folder', type=str, default='caption/data/coco/img')
parser.add_argument('--json_path', type=str, default='caption/data/coco/dataset_coco.json')
config = parser.parse_args()

# class Config(object):
#     def __init__(self):
#         self.train_batch_size = 5
#         self.val_batch_size = 5
#         self.test_batch_size = 5
#         self.image_folder = 'caption/data/coco/img'
#         self.json_path = 'caption/data/coco/dataset_coco.json'
#         self.max_dec_len = 50
# config = Config()

data_loader = DataLoader(config)
data_loader.parse_json()
data_loader.construct_vocab()


start = time.time()
a, b, c, d = data_loader.next_batch(mode='train')
end = time.time()
print(end-start)

visualize(a, c)
