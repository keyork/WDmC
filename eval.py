import torch


def test(dataloader, model, device, loss_fn):
    print("Test! ", end="")
    size = len(dataloader.dataset)
    dts = 0
    loss = 0
    corr = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            result = 1.0 * (pred >= 0.5)
            corr += (1 * (((result == y).sum(axis=1)) == result.shape[1])).sum()
            batch_loss = loss_fn(pred, y.float())
            hamming_dts = torch.abs(result - y).sum() / 8
            dts += hamming_dts
            loss += batch_loss
        print(
            "hamming_dts={:5f}, loss={:5f}, acc={:5f}".format(
                dts / size, loss / size, corr / size
            )
        )
