import torch
import time

def metric_batch(output, target):
    pred = output.argmax(1)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt = None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b

def loss_epoch(model, loss_func, dataloader, device, opt = None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataloader)

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)

        loss_b, metric_b = loss_batch(loss_func, output, labels, opt = opt)
        running_loss += loss_b
        if opt is not None:
            running_metric += metric_b

        loss = running_loss / len_data
        metric = running_metric / len_data

    return loss, metric


def train_and_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        print('EPOCH : ', epoch, 'LOSS : {:.4f}'.format(train_loss), 'METRIC : {:.3f}'.format(train_metric))

        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                print('Have a best Model')
                print('EPOCH : ', epoch, 'LOSS : {:.4f}'.format(val_loss), 'METRIC : {:.3f}'.format(val_metric))

    return model, loss_history, metric_history
