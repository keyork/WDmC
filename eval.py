import torch
import numpy as np
from tqdm import tqdm


def test(dataloader, model, device, loss_fn, is_neck):
    print("Test! ", end="")
    size = len(dataloader.dataset)
    dts = 0
    loss = 0
    corr = 0
    model.eval()
    if not is_neck:
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
    else:
        with torch.no_grad():
            for batch, (X1, X2, y) in enumerate(dataloader):
                X1, X2, y = X1.to(device), X2.to(device), y.to(device)
                X = {"raw": X1, "resize": X2}

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


def get_result_file(dataloader, model, device, target_path, is_neck):
    print("Test!")
    model.eval()
    if not is_neck:
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(dataloader)):
                X, y = X.to(device), y.to(device)
                # Compute prediction error
                pred = model(X)
                result = 1.0 * (pred >= 0.5)
                if device == "cuda":
                    result = result.cpu()
                    X = X.cpu()
                result = result.numpy()
                X = X.numpy()
                print(X.shape)
                if batch == 0:
                    final_result = result
                    final_data = X
                else:
                    final_result = np.vstack((final_result, result))
                    final_data = np.vstack((final_data, X))
    else:
        with torch.no_grad():
            for batch, (X1, X2, y) in tqdm(enumerate(dataloader)):
                X1, X2, y = X1.to(device), X2.to(device), y.to(device)
                X = {"raw": X1, "resize": X2}
                # Compute prediction error
                pred = model(X)
                result = 1.0 * (pred >= 0.5)
                if device == "cuda":
                    result = result.cpu()
                    X = X.cpu()
                result = result.numpy()
                X = X.numpy()
                print(X.shape)
                if batch == 0:
                    final_result = result
                    final_data = X
                else:
                    final_result = np.vstack((final_result, result))
                    final_data = np.vstack((final_data, X))
    np.savez(target_path, test=final_data, result=final_result)


def get_result_file_raw(dataloader, model, device, target_path, is_neck):
    print("Test!")
    model.eval()
    test_num = len(dataloader)
    if is_neck:
        for idx in tqdm(range(test_num), ncols=70):
            X = dataloader[idx]
            X["raw"] = X["raw"].to(device)
            X["resize"] = X["resize"].to(device)
            pred = model(X)
            result = 1.0 * (pred >= 0.5)
            if device == "cuda":
                result = result.cpu()
            result = result.numpy()
            if idx == 0:
                final_result = result
            else:
                final_result = np.vstack((final_result, result))
    np.savetxt(target_path, final_result, fmt="%d", delimiter=",")
