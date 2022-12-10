import torch


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_fn,
    device,
    writer_group,
    epoch_idx,
    scheduler,
):

    # Train
    valid_dts = 0
    valid_times = 0
    batch_gap = 20

    train_dts = 0
    train_corr = 0
    train_loss = 0

    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        model.train()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get train acc & hamming dts
        train_result = 1.0 * (pred >= 0.5)
        train_corr += (
            1 * (((train_result == y).sum(axis=1)) == train_result.shape[1])
        ).sum()
        train_hamming_dts = torch.abs(train_result - y).sum() / 8

        train_dts += train_hamming_dts
        train_loss += loss.item()

        if (batch + 1) % batch_gap == 0:

            loss, current = train_loss / (batch + 1) * len(X), batch * len(X)

            print(
                f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] lr: {optimizer.state_dict()['param_groups'][0]['lr']:>9f}"
            )

            # draw curves
            writer_group["loss"].add_scalar(
                "training_loss",
                train_loss / batch_gap * len(X),
                epoch_idx * size + (batch + 1) * len(X),
            )
            writer_group["acc"].add_scalar(
                "training_acc",
                train_corr / batch_gap * len(X),
                epoch_idx * size + (batch + 1) * len(X),
            )
            writer_group["dts"].add_scalar(
                "training_dts",
                train_dts / batch_gap * len(X),
                epoch_idx * size + (batch + 1) * len(X),
            )

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
                writer_group["loss"].add_scalar(
                    "valid_loss", loss / val_size, epoch_idx * size + batch
                )
                writer_group["acc"].add_scalar(
                    "valid_acc", corr / val_size, epoch_idx * size + batch
                )
                writer_group["dts"].add_scalar(
                    "valid_dts", dts / val_size, epoch_idx * size + batch
                )
                valid_dts += dts
                valid_times += 1
            model.train()
            train_dts = 0
            train_corr = 0
            train_loss = 0
    scheduler.step(valid_dts / valid_times)
