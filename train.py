from tqdm import tqdm
from network import LatentModel
from torch.utils.data import Dataset, DataLoader

import torch as t
from torch.utils.data import DataLoader
from pair_encoder import TPair
import argparse
import os
import json
import random
import time

DEVICE = 'cuda' if t.cuda.is_available() else "cpu"
DB = os.getenv('DB')

# Define Dataset Loader Here
class CustomDataset(Dataset):
    def __init__(self, pairs_ctx, elapsed_ctx, pairs_tar, elapsed_tar):
        self.pairs_ctx = pairs_ctx
        self.elapsed_ctx = elapsed_ctx
        self.pairs_tar = pairs_tar
        self.elapsed_tar = elapsed_tar

    def __getitem__(self, index):
        return self.pairs_ctx[index], self.elapsed_ctx[index], \
            self.pairs_tar[index], self.elapsed_tar[index]

    def __len__(self):
        return len(self.pairs_ctx)

def load_dataset(path, data_size=300):
    pairs, elapsed = [], []
    with open(path) as f:
        line = f.readline()
        while line:
            items = line.split('\t')
            stats = json.loads(items[3])
            if len(stats) > 0:
                pairs.append(TPair(json.loads(items[1])['plan'], json.loads(items[2])))
                elapsed.append([stats['elapsed'], stats['fail']])
            line = f.readline()
    
    item_len = len(pairs)
    ctx_num = item_len // 2
    tar_num = item_len // 4
    print("read dataset, size %d" % item_len)
    pairs_ctx, elapseds_ctx, pairs_tar, elapseds_tar = [], [], [], []
    for _ in range(data_size):
        pair_ctx, elapsed_ctx, pair_tar, elapsed_tar = [], [], [], []
        for i in range(ctx_num):
            idx = random.choice(range(item_len))
            pair_ctx.append(pairs[idx])
            elapsed_ctx.append(elapsed[idx])
        for i in range(tar_num):
            idx = random.choice(range(item_len))
            pair_tar.append(pairs[idx])
            elapsed_tar.append(elapsed[idx])
        pairs_ctx.append(pair_ctx)
        elapseds_ctx.append(t.Tensor(elapsed_ctx))
        pairs_tar.append(pair_tar)
        elapseds_tar.append(t.Tensor(elapsed_tar))
    
    return CustomDataset(pairs_ctx, elapseds_ctx, pairs_tar, elapseds_tar)


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def verify(dict_name):
    dataset = load_dataset(f"data/{DB}_samples", 1)
    dloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=True)
    for _, data in enumerate(dloader):
        x1, y1, x2, y2 = data
        state_dict = t.load(dict_name)
        model = LatentModel(32).to(DEVICE)
        model.load_state_dict(state_dict['model'])
        model.train(False)
        start = time.time()
        res, status, loss = model(x1, y1, x2, None)
        print("cost %f" % (time.time() - start))
        latency = y2[:, 0].unsqueeze(-1)
        print(y2, t.cat([res.stddev, res.mean], dim=1))
        mae = t.abs(res.mean - latency).mean()
        print("mae %f" % mae)
        break

        
def main(file_name, output_path, epochs, lr):
    model = LatentModel(32).to(DEVICE)
    model.train()
    
    optim = t.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    train_dataset = load_dataset(file_name)
    for epoch in range(epochs):
        dloader = DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=True)
        pbar = tqdm(dloader)
        loss_accum = 0.
        for i, data in enumerate(pbar):
            global_step += 1
            adjust_learning_rate(optim, global_step)
            x1, y1, x2, y2 = data
            
            # pass through the latent model
            y_pred_l, y_pred_e, loss = model(x1, y1, x2, y2)
            loss_accum += loss
            
            # Training step
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print("loss : {} kl: {}".format(loss.mean(), kl.mean()) ) 
            # Logging
            # writer.add_scalars('training_loss',{
            #         'loss':loss,
            #         'kl':kl.mean(),

            #     }, global_step)
            
        # save model by each epoch    
        t.save({'model':model.state_dict(),
                                 'optimizer':optim.state_dict()},
                                os.path.join(output_path))
        print("loss %f" % (loss_accum / len(train_dataset)))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Knob-Plan Encoder & Dual-Task Predictor')
    parser.add_argument('--sample_file', type=str, required=True,
                        help='specifies the file that contains sampled data')
    parser.add_argument('--model_output', type=str, required=True,
                        help='specifies the path where the trained model is output')
    parser.add_argument('--epoch', type=int, required=True,
                        help='specifies training epochs')
    parser.add_argument('--lr', type=float, required=True,
                        help='specifies the learning rate of training')

    args = parser.parse_args()
    
    main(f"data/{args.sample_file}", args.model_output, args.epoch, args.lr)
    # verify("checkpoint/stats.pth.tar")
