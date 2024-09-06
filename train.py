import torch

from darknet import Darknet
from dataloader import OpenDataset, df, DATA_PATH, _transform, DEVICE
from model import CONFIG_PATH


def train_batch(model, data, optimizer, criterion):
    """
    _train the model on a batch of data
    :return: _value of loss and accuracy.
    """
    model.train()
    images, boxes = data
    out_box = model(images)
    optimizer.zero_grad()

    loss, acc = criterion(out_box, boxes)
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    """
    _validate the model on a batch of data
    :return: _value of loss and accuracy.
    """
    model.eval()
    images, boxes = data
    out_box = model(images)
    loss, acc = criterion(out_box, boxes)

    return loss.item(), acc.item()


if __name__ == '__main__':
    train_dataset = OpenDataset(df, DATA_PATH + 'training_images/', _transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn)
    net = Darknet(CONFIG_PATH).to(DEVICE)

    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        N = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            loss, acc = train_batch(net, data, optimizer, criterion)
            print(f'Epoch: {epoch}, Batch: {i}/{N}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        if epoch % 10 == 0:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    torch.save(net.state_dict(), 'model.pth')
