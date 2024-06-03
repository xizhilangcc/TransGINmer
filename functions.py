import torch
import os
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import metric
from utils import *

def save_weights(model,epoch,folder):
    if os.path.isdir(folder)==False:
        os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder+'/num_train{}.ckpt'.format(epoch+1))



def validate(model,device,dataset, min_length, threshold,nprocesss=1, batch_size=128):
    batches=len(dataset)
    model.train(False)
    total=0
    predictions=[]
    outputs=[]
    ground_truths=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():

        # accuracy, sensitivity, specificity = evaluate_model(model, sequences_and_labels, min_length, device,threshold)


        for data in tqdm(dataset,ncols=80):
            X=data['data'].to(device)
            Y=data['labels'].to(device)
            Y_int64 = Y.to(torch.int64)

            output= model(X,threshold)
            del X
            # 将模型输出转换为类别概率
            output = output.float()
            loss+=criterion(output,Y_int64)
            output = F.softmax(output, dim=1)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in classification_predictions:
                predictions.append(pred.cpu().numpy())
            for vector in output:
                outputs.append(vector.cpu().numpy())
            for t in Y:
                ground_truths.append(t.cpu().numpy())
            del output
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    ground_truths=np.asarray(ground_truths)
    predictions=np.asarray(predictions)
    outputs=np.asarray(outputs)
    #print(predictions)
    #print(ground_truths)
    #score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    val_acc=metric.accuracy(predictions,ground_truths)
    auc=metrics.roc_auc_score(ground_truths,outputs[:,1])
    val_sens=metric.sensitivity(predictions,ground_truths)
    val_spec=metric.specificity(predictions,ground_truths)
    print('Val accuracy: {}, Val Loss: {}'.format(val_acc,val_loss))
    return val_loss,auc,val_acc,val_sens,val_spec
    # return val_loss,auc,val_acc,val_sens,val_spec

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_item = torch.from_numpy(self.sequences.iloc[idx].values)
        label_item = torch.tensor(self.labels.iloc[idx])
        return {'data': data_item, 'labels': label_item}