# two ways


class Config(object):

    def __init__(self):

        self.train_batch_size = 64
        self.val_batch_size = 5
        self.test_batch_size = 32
        self.max_dec_len = 50
        self.encoded_image_size = 14
        self.image_folder = 'caption/data/coco/img'
        self.json_path = 'caption/data/coco/dataset_coco.json'
        self.encoder_dim = 2048
        self.decoder_dim = 512
        self.attention_dim = 512
        self.embed_dim = 512
        self.dropout = 0.3
        self.device = 'cuda'


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--max_dec_len', type=int, default=50)
    parser.add_argument('--encoded_image_size', type=int, default=7)
    parser.add_argument('--encoder_dim', type=int, default=2048)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)

    parser.add_argument('--image_folder', type=str, default='caption/data/coco/img')
    parser.add_argument('--json_path', type=str, default='caption/data/coco/dataset_coco.json')

    config = parser.parse_args()

