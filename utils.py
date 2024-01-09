import os
import time
import random
import numpy as np
import pandas as pd
from pprint import pprint

import torch
import torch.nn as nn

from . import models
from .datasets import get_obese3d_loaders



def train(device, model, data_loader, criterion, optimizer, scheduler=None):
    model = model.to(device)

    model.train()
    total_loss = 0.0
    correct_pred = 0
    correct_top5_pred = 0
    total_pred = 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_pred += targets.size(0)
        correct_pred += (predicted == targets).sum().item()

        # Calculate top-5 accuracy
        _, top5_pred = outputs.topk(5, 1, True, True)
        correct_top5_pred += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    avg_loss = total_loss / len(data_loader)
    avg_acc = correct_pred / total_pred
    avg_top5_acc = correct_top5_pred / total_pred
    return avg_loss, avg_acc, avg_top5_acc

def evaluate(device, model, data_loader, criterion):
    with torch.no_grad():
        model = model.to(device)

        model.eval()
        total_loss = 0.0
        correct_pred = 0
        correct_top5_pred = 0
        total_pred = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_pred += targets.size(0)
            correct_pred += (predicted == targets).sum().item()

            # Calculate top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5_pred += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

        avg_loss = total_loss / len(data_loader)
        avg_acc = correct_pred / total_pred
        avg_top5_acc = correct_top5_pred / total_pred
        return avg_loss, avg_acc, avg_top5_acc

def benchmark(args):
    # Convert args to dictionary
    args_dict = vars(args)

    # Save the dictionary to a JSON file
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        json_path = os.path.join( args.save_dir, 'args.json')
        with open(json_path, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)
    
    pprint(args_dict)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    input_shape = [args.batch_size, args.seq_len, args.num_joints, args.dimension]
    hidden_size = args.hidden_size
    output_size = args.output_size

    train_loader, valid_loader, test_loader = get_obese3d_loaders(data_dir=args.data_dir, batch_size=args.batch_size, target_type=args.target_type, seq_len=args.seq_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, args.model_name)(input_shape, hidden_size, output_size, **args.model_kwrags)
    criterion = getattr(nn, args.criterion_name)(**args.criterion_kwargs)
    optimizer = getattr(torch.optim, args.optimizer_name)(model.parameters(), lr=args.lr, **args.optimizer_kwargs)

    logs = []
    total_start_time = time.time()

    best_loss, best_acc, best_top5_acc = np.inf, 0, 0
    for epoch in range(1, 1+args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc, train_top5_acc = train(device, model, train_loader, criterion, optimizer)
        valid_loss, valid_acc, valid_top5_acc = evaluate(device, model, valid_loader, criterion)

        if valid_loss < best_loss: 
            best_loss = valid_loss

        if valid_acc > best_acc: 
            best_acc = valid_acc
            if args.save_dir:
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_state_dict.pth') )

        if valid_top5_acc > best_top5_acc: 
            best_top5_acc = valid_top5_acc
        
        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - total_start_time

        log = dict(
            epoch=epoch, epoch_duration=epoch_duration, total_duration=total_duration,
            train_loss=train_loss, train_acc=train_acc, train_top5_acc=train_top5_acc, 
            valid_loss=valid_loss, valid_acc=valid_acc, valid_top5_acc=valid_top5_acc,
            best_loss=best_loss, best_acc=best_acc, best_top5_acc=best_top5_acc)
        logs.append( log )
        
        if args.save_dir:
            df = pd.DataFrame(logs)
            df.to_csv( os.path.join(args.save_dir, 'log.csv') )

        if not args.quiet:
            msg = f'[Epoch {epoch:3}/{args.epochs:3} {epoch_duration:6.2f}s {total_duration/60:6.2f}m] Metric: train/valid[best] | Loss: {train_loss:5.3f}/{valid_loss:5.3f}[{best_loss:5.3f}] '\
            f'| Top-1 Acc.: {train_acc:6.2%}/{valid_acc:6.2%}[{best_acc:6.2%}] | Top-5 Acc.: {train_top5_acc:6.2%}/{valid_top5_acc:6.2%}[{best_top5_acc:6.2%}]'
            print(msg)