import argparse
import torch
from model import TransGIN                                                 
from Logger import *
from functions import *





def prediction_fasta(model, sequences_and_labels, length, device,threshold1):
    results = [] 

    for seq_id, sampled_seqs in tqdm(sequences_and_labels.items()):
        predictions_list = []
        batches = create_labeled_feature_vectors_with_sorted_codons6(sampled_seqs,label=1)
        inputs = torch.tensor(batches, device=device)
        with torch.no_grad():
            for input_batch in inputs:
                predictions = model(input_batch.unsqueeze(0),threshold1)
                predictions_list.append(predictions)
        average_predictions = torch.mean(torch.stack(predictions_list), dim=0)
        average_predictions = torch.softmax(average_predictions, dim=1)
        threshold = 0.5
        binary_predictions = 1 if average_predictions[:, 1] >= threshold else 0
        scores = average_predictions[:, 1].detach().cpu().numpy()
        scores = str(scores[0])

        results.append((seq_id, binary_predictions,scores))

    with open('predictions.txt', 'w') as f:
        for seq_id, prediction,scores in results:
            f.write(f'{seq_id}\t{prediction}\t{scores}\n')

    print("File 'predictions.txt' has been created.")



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='contig.fasta')
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_heads', type=float, default=2)
    parser.add_argument('--num_classes', type=float, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=733)
    parser.add_argument('--lr_scale', type=float, default=0.1)
    parser.add_argument('--num_node_features', type=int, default=64)
    parser.add_argument('--min_length', type=int, default=998)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--atten_num_layers', type=int, default=1)
    parser.add_argument('--atten_dim_feedforward', type=int, default=4)
    parser.add_argument('--gin_num_layers', type=int, default=1)
    parser.add_argument('--gin_dim_feedforward', type=int, default=2)

    opts = parser.parse_args()
    return opts


opts = get_args()
min_length = opts.min_length
threshold1 = opts.threshold



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("running with cpu")



digital_prediction_fasta = translate2digital_sequence(opts.dataset, window_size=3, stride=1)



model = TransGIN(opts.num_node_features, opts.hidden_channels, opts.num_classes,
                     opts.num_heads, opts.atten_num_layers, opts.atten_dim_feedforward,opts.gin_num_layers,opts.gin_dim_feedforward).to(device)

checkpoints_folder='model.ckpt'

state_dict = torch.load(checkpoints_folder, map_location=device)
model.load_state_dict(state_dict)
model.train(False)

prediction_fasta(model, digital_prediction_fasta, 998, device,threshold1)

