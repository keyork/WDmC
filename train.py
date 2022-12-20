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
    is_neck,
):

    # Train
    valid_dts = 0
    valid_times = 0
    batch_gap = 20

    train_dts = 0
    train_corr = 0
    train_loss = 0

    size = len(train_loader.dataset)
    if not is_neck:
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

                loss, current = train_loss / (len(X) * batch_gap), batch * len(X)

                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] lr: {optimizer.state_dict()['param_groups'][0]['lr']:>9f}"
                )

                # draw curves
                writer_group["train"].add_scalar(
                    "loss",
                    train_loss / (len(X) * batch_gap),
                    epoch_idx * size + (batch + 1) * len(X),
                )
                writer_group["train"].add_scalar(
                    "acc",
                    train_corr / (len(X) * batch_gap),
                    epoch_idx * size + (batch + 1) * len(X),
                )
                writer_group["train"].add_scalar(
                    "dts",
                    train_dts / (len(X) * batch_gap),
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
                    for batch_val, (X, y) in enumerate(valid_loader):
                        X, y = X.to(device), y.to(device)

                        # Compute prediction error
                        pred = model(X)
                        result = 1.0 * (pred >= 0.5)
                        corr += (
                            1 * (((result == y).sum(axis=1)) == result.shape[1])
                        ).sum()

                        batch_loss = loss_fn(pred, y.float())
                        hamming_dts = torch.abs(result - y).sum() / 8
                        dts += hamming_dts
                        loss += batch_loss.item()
                    print(
                        "hamming_dts={:5f}, loss={:5f}, acc={:5f}".format(
                            dts / val_size, loss / val_size, corr / val_size
                        )
                    )
                    writer_group["valid"].add_scalar(
                        "loss", loss / val_size, epoch_idx * size + batch_val
                    )
                    writer_group["valid"].add_scalar(
                        "acc", corr / val_size, epoch_idx * size + batch_val
                    )
                    writer_group["valid"].add_scalar(
                        "dts", dts / val_size, epoch_idx * size + batch_val
                    )
                    valid_dts += dts
                    valid_times += 1
                model.train()
                train_dts = 0
                train_corr = 0
                train_loss = 0
        scheduler.step(valid_dts / valid_times)
    else:
        for batch, (X1, X2, y) in enumerate(train_loader):
            model.train()
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            X = {"raw": X1, "resize": X2}
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

                loss, current = train_loss / (len(X1) * batch_gap), batch * len(X1)

                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] lr: {optimizer.state_dict()['param_groups'][0]['lr']:>9f}"
                )

                # draw curves
                writer_group["train"].add_scalar(
                    "loss",
                    train_loss / (len(X1) * batch_gap),
                    epoch_idx * size + (batch + 1) * len(X1),
                )
                writer_group["train"].add_scalar(
                    "acc",
                    train_corr / (len(X1) * batch_gap),
                    epoch_idx * size + (batch + 1) * len(X1),
                )
                writer_group["train"].add_scalar(
                    "dts",
                    train_dts / (len(X1) * batch_gap),
                    epoch_idx * size + (batch + 1) * len(X1),
                )

                # Valid

                print("Valid! ", end="")
                val_size = len(valid_loader.dataset)
                dts = 0
                loss = 0
                corr = 0
                model.eval()
                with torch.no_grad():
                    for batch_val, (X1, X2, y) in enumerate(valid_loader):
                        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
                        X = {"raw": X1, "resize": X2}
                        # Compute prediction error
                        pred = model(X)
                        result = 1.0 * (pred >= 0.5)
                        corr += (
                            1 * (((result == y).sum(axis=1)) == result.shape[1])
                        ).sum()

                        batch_loss = loss_fn(pred, y.float())
                        hamming_dts = torch.abs(result - y).sum() / 8
                        dts += hamming_dts
                        loss += batch_loss.item()
                    print(
                        "hamming_dts={:5f}, loss={:5f}, acc={:5f}".format(
                            dts / val_size, loss / val_size, corr / val_size
                        )
                    )
                    writer_group["valid"].add_scalar(
                        "loss", loss / val_size, epoch_idx * size + batch_val
                    )
                    writer_group["valid"].add_scalar(
                        "acc", corr / val_size, epoch_idx * size + batch_val
                    )
                    writer_group["valid"].add_scalar(
                        "dts", dts / val_size, epoch_idx * size + batch_val
                    )
                    valid_dts += dts
                    valid_times += 1
                model.train()
                train_dts = 0
                train_corr = 0
                train_loss = 0
        scheduler.step(valid_dts / valid_times)
