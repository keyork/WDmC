import torch

def train(model, train_loader, valid_loader, optimizer, loss_fn, device):
    
    # Train
    
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # Valid
            
            print("Valid! ", end="")
            val_size = len(valid_loader.dataset)
            dts = 0
            loss = 0
            corr = 0
            model.eval()
            with torch.no_grad():
                for batch, (X, y) in enumerate(valid_loader):
                    X, y = X.to(device), y.to(device)

                    # Compute prediction error
                    pred = model(X)
                    result = 1.0 * (pred >= 0.5)
                    corr += (1 * (((result == y).sum(axis=1)) == result.shape[1])).sum()
                    # batch_loss = cross_entropy_one_hot(result, y)
                    batch_loss = loss_fn(pred, y.float())
                    hamming_dts = torch.abs(result - y).sum() / 8
                    dts += hamming_dts
                    loss += batch_loss
                print(
                    "hamming_dts={:5f}, loss={:5f}, acc={:5f}".format(
                        dts / val_size, loss / val_size, corr / val_size
                    )
                )
    