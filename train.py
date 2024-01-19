import torch.nn.functional as F

def train(model, device, optimizer, scheduler, data_loader):
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i * len(data), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item()))