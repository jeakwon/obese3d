import torch

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
